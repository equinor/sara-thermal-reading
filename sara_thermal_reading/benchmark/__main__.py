import argparse
import platform
import re
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.benchmark.storage import BenchmarkStorage
from sara_thermal_reading.benchmark.thermal_metrics import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    PER_IMAGE_COLUMNS,
    PROTOCOL,
    ThermalBenchmarkMetrics,
    rasterize_polygon,
)
from sara_thermal_reading.benchmark.thermal_split import ThermalSplit
from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.image_alignment.align_dispatcher import (
    SUPPORTED_ALIGNMENT_METHODS,
    AlignmentMethod,
    align_thermal_images,
)
from sara_thermal_reading.image_processing.find_temperature_in_polygon import (
    find_temperature_in_polygon,
)
from sara_thermal_reading.main_thermal_workflow import (
    preprocess_thermal_pair_for_alignment,
)

SCHEMA_VERSION = 3
PREDICTIONS_SCHEMA_VERSION = 1


def evaluate_split(
    split: ThermalSplit,
    method: AlignmentMethod,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[
    dict[str, Any],
    ThermalBenchmarkMetrics,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics = ThermalBenchmarkMetrics(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    per_image_items: list[dict[str, Any]] = []
    prediction_items: list[dict[str, Any]] = []
    reference_cache: dict[str, NDArray[np.float64]] = {}
    elapsed = 0.0

    for sample_id in sorted(split.samples):
        sample = split.samples[sample_id]
        reference = split.references[sample.reference_id]
        if sample.reference_id not in reference_cache:
            reference_cache[sample.reference_id] = split.read_reference_tiff(
                sample.reference_id
            )
        reference_image = reference_cache[sample.reference_id]
        source_image = split.read_source_tiff(sample_id)

        started = perf_counter()
        ref_uint8, src_uint8 = preprocess_thermal_pair_for_alignment(
            reference_image, source_image
        )
        alignment = align_thermal_images(
            ref_uint8,
            src_uint8,
            reference.polygon,
            method=method,
        )
        pred_temp_c = _safe_polygon_temperature(
            source_image,
            alignment.warped_polygon,
            settings.TEMPERATURE_PERCENTILE,
        )
        elapsed += perf_counter() - started

        gt_temp_c = (
            _safe_polygon_temperature(
                source_image,
                sample.gt_polygon,
                settings.TEMPERATURE_PERCENTILE,
            )
            if sample.matchable and sample.gt_polygon is not None
            else None
        )

        sample_eval = metrics.add_sample(
            matchable=sample.matchable,
            width=sample.width,
            height=sample.height,
            gt_polygon=sample.gt_polygon,
            predicted_polygon=alignment.warped_polygon,
            matching_confidence=alignment.matching_confidence,
            gt_temp_c=gt_temp_c,
            pred_temp_c=pred_temp_c,
        )

        per_image_items.append(
            {
                "image_id": sample.sample_id,
                "file_name": sample.file_name,
                "width": sample.width,
                "height": sample.height,
                "passed": sample_eval.passed,
                "missed": sample_eval.missed,
                "false_positive": sample_eval.false_positive,
                "values": {
                    "matchable": sample_eval.matchable,
                    "accepted": sample_eval.accepted,
                    "confidence": sample_eval.confidence,
                    "polygon_iou": sample_eval.polygon_iou,
                    "centroid_error_px": sample_eval.centroid_error_px,
                    "gt_temp_c": sample_eval.gt_temp_c,
                    "pred_temp_c": sample_eval.pred_temp_c,
                    "temp_error_c": sample_eval.temp_error_c,
                },
            }
        )
        prediction_items.append(
            {
                "sample_id": sample.sample_id,
                "predicted_polygon": [
                    [float(x), float(y)] for x, y in alignment.warped_polygon
                ],
                "predicted_temperature": pred_temp_c,
                "gt_temperature": gt_temp_c,
                "temperature_error_c": sample_eval.temp_error_c,
                "polygon_iou": sample_eval.polygon_iou,
                "centroid_error_px": sample_eval.centroid_error_px,
                "matching_confidence": alignment.matching_confidence,
                "raw_score": alignment.raw_score,
                "accepted": sample_eval.accepted,
            }
        )

    count = len(split.samples)
    processing = {
        "unit": "image",
        "count": count,
        "seconds": elapsed,
        "per_second": count / elapsed if elapsed > 0.0 else 0.0,
    }
    return processing, metrics, per_image_items, prediction_items


def _safe_polygon_temperature(
    thermal_image: NDArray[np.float64],
    polygon: list[tuple[int, int]] | list[tuple[float, float]],
    percentile: float,
) -> float | None:
    height, width = thermal_image.shape[:2]
    mask = rasterize_polygon(polygon, width=width, height=height)
    if not np.any(mask):
        return None
    polygon_array = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    return find_temperature_in_polygon(thermal_image, polygon_array, percentile)


def hardware_name() -> str:
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine()


def path_segment(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise argparse.ArgumentTypeError("Expected a single path segment")
    return value


def parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark thermal reading alignment")
    parser.add_argument(
        "--source", required=True, help="Dataset directory or container URL"
    )
    parser.add_argument(
        "--destination", required=True, help="Results directory or container URL"
    )
    parser.add_argument("--version", required=True, type=path_segment)
    parser.add_argument("--split", default="val", type=path_segment)
    parser.add_argument("--run-id", required=True, type=path_segment)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument(
        "--method",
        default="phase_correlation",
        choices=SUPPORTED_ALIGNMENT_METHODS,
    )
    parser.add_argument(
        "--artifact", required=True, help="Exact thermal-reading image or build ID"
    )
    parser.add_argument(
        "--confidence-threshold",
        default=None,
        type=parse_optional_float,
        help="Minimum matching confidence threshold to accept an alignment",
    )
    parser.add_argument(
        "--iou-threshold",
        default=None,
        type=parse_optional_float,
        help="Minimum polygon IoU required to count an alignment as valid",
    )
    return parser


def write_benchmark_outputs(
    destination: BenchmarkStorage,
    prefix: str,
    args: argparse.Namespace,
    source_name: str,
    split: ThermalSplit,
    method: AlignmentMethod,
    confidence_threshold: float,
    iou_threshold: float,
    processing: dict[str, Any],
    metrics: ThermalBenchmarkMetrics,
    per_image_items: list[dict[str, Any]],
    prediction_items: list[dict[str, Any]],
) -> None:
    common = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id}
    destination.create_json(
        f"{prefix}/run.json",
        {
            **common,
            "dataset": {
                "id": split.dataset_id or source_name,
                "version": args.version,
                "split": args.split,
                "manifest_sha256": split.manifest_sha256,
            },
            "artifact": args.artifact,
            "evaluator": args.artifact,
            "protocol": PROTOCOL,
            "configuration": {
                "method": method,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "temperature_percentile": settings.TEMPERATURE_PERCENTILE,
            },
            "execution": {
                "hardware": hardware_name(),
                "timing_scope": "inference_only",
            },
        },
    )
    destination.create_json(
        f"{prefix}/metrics.json",
        {
            **common,
            "processing": processing,
            "summary_metrics": metrics.as_summary_metrics(),
            "evaluations": metrics.as_evaluations(),
        },
    )
    destination.create_json(
        f"{prefix}/review/per_image.json",
        {
            "columns": PER_IMAGE_COLUMNS,
            "items": per_image_items,
        },
    )
    destination.create_json(
        f"{prefix}/review/predictions.json",
        {
            "schema_version": PREDICTIONS_SCHEMA_VERSION,
            "run_id": args.run_id,
            "items": prediction_items,
        },
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    confidence_threshold = (
        DEFAULT_CONFIDENCE_THRESHOLD
        if args.confidence_threshold is None
        else args.confidence_threshold
    )
    if not (0.0 <= confidence_threshold <= 1.0):
        parser.error("confidence-threshold must be in [0.0, 1.0]")

    iou_threshold = (
        DEFAULT_IOU_THRESHOLD if args.iou_threshold is None else args.iou_threshold
    )
    if not (0.0 < iou_threshold <= 1.0):
        parser.error("iou-threshold must be in (0.0, 1.0]")

    method = cast(AlignmentMethod, args.method)
    source = BenchmarkStorage(args.source)
    destination = BenchmarkStorage(args.destination)
    prefix = f"{args.version}/benchmark-results/{args.split}/{args.run_id}"
    if any(
        destination.exists(f"{prefix}/{name}") for name in ("run.json", "metrics.json")
    ):
        raise FileExistsError(f"Benchmark run already exists: {prefix}")

    split = ThermalSplit.load(source.read, args.version, args.split)
    processing, metrics, per_image_items, prediction_items = evaluate_split(
        split=split,
        method=method,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    write_benchmark_outputs(
        destination=destination,
        prefix=prefix,
        args=args,
        source_name=source.name,
        split=split,
        method=method,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        processing=processing,
        metrics=metrics,
        per_image_items=per_image_items,
        prediction_items=prediction_items,
    )


if __name__ == "__main__":
    main()
