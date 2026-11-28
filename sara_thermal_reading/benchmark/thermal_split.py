import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.file_io.tiff_loader import load_thermal_tiff_from_bytes

CONTAINER_SCHEMA_FILENAME = "schema.json"
THERMAL_DATASET_FORMAT = "thermal"
THERMAL_SCHEMA_VERSION = 1

Polygon = list[tuple[float, float]]


@dataclass(frozen=True)
class ThermalReference:
    reference_id: str
    installation_code: str
    tag_id: str
    inspection_description: str
    tiff_file: str
    preview_file: str
    width: int
    height: int
    polygon: Polygon


@dataclass(frozen=True)
class ThermalSample:
    sample_id: int
    file_name: str
    tiff_file: str
    preview_file: str
    width: int
    height: int
    reference_id: str
    matchable: bool
    gt_polygon: Polygon | None


@dataclass
class ThermalSplit:
    read_container: Callable[[str], bytes]
    split: str
    manifest_sha256: str
    dataset_id: str | None
    references: dict[str, ThermalReference]
    samples: dict[int, ThermalSample]

    @classmethod
    def load(
        cls,
        read_container: Callable[[str], bytes],
        version: str,
        split: str,
    ) -> "ThermalSplit":
        validate_container_schema(read_container(CONTAINER_SCHEMA_FILENAME))
        dataset_prefix = f"{version}/dataset"

        manifest_bytes = read_container(f"{dataset_prefix}/manifest.json")
        dataset_id = parse_and_validate_manifest(manifest_bytes)
        annotations_data = json.loads(
            read_container(f"{dataset_prefix}/annotations/samples_{split}.json")
        )
        references, samples = parse_thermal_samples_file(annotations_data)
        return cls(
            read_container=read_container,
            split=split,
            manifest_sha256=sha256(manifest_bytes).hexdigest(),
            dataset_id=dataset_id,
            references=references,
            samples=samples,
        )

    def read_reference_tiff(self, reference_id: str) -> NDArray[np.float64]:
        ref = self.references[reference_id]
        validate_relative_path(ref.tiff_file, f"reference {reference_id} tiff_file")
        image = load_thermal_tiff_from_bytes(self.read_container(ref.tiff_file))
        _verify_dimensions(image, ref.width, ref.height, f"reference {reference_id}")
        return image

    def read_source_tiff(self, sample_id: int) -> NDArray[np.float64]:
        sample = self.samples[sample_id]
        validate_relative_path(sample.tiff_file, f"sample {sample_id} tiff_file")
        image = load_thermal_tiff_from_bytes(self.read_container(sample.tiff_file))
        _verify_dimensions(image, sample.width, sample.height, f"sample {sample_id}")
        return image


def validate_container_schema(schema_bytes: bytes) -> None:
    data = json.loads(schema_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {CONTAINER_SCHEMA_FILENAME}")
    dataset_format = data.get("format")
    if dataset_format != THERMAL_DATASET_FORMAT:
        raise ValueError(
            f"Unsupported dataset format '{dataset_format}' in {CONTAINER_SCHEMA_FILENAME}; "
            f"expected '{THERMAL_DATASET_FORMAT}'"
        )
    schema_version = data.get("schema_version")
    if schema_version != THERMAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version '{schema_version}' in {CONTAINER_SCHEMA_FILENAME}; "
            f"expected {THERMAL_SCHEMA_VERSION}"
        )


def parse_and_validate_manifest(manifest_bytes: bytes) -> str | None:
    manifest_data = json.loads(manifest_bytes)
    if not isinstance(manifest_data, dict):
        raise ValueError("Expected a JSON object in manifest.json")
    manifest_format = manifest_data.get("format")
    if manifest_format != THERMAL_DATASET_FORMAT:
        raise ValueError(
            f"Unsupported dataset format '{manifest_format}' in manifest.json; "
            f"expected '{THERMAL_DATASET_FORMAT}'"
        )
    for key in ("dataset_id", "dataset", "id"):
        value = manifest_data.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Expected a non-empty string for {key} in manifest.json"
                )
            return value.strip()
    return None


def parse_thermal_samples_file(
    data: dict[str, Any],
) -> tuple[dict[str, ThermalReference], dict[int, ThermalSample]]:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object in thermal samples annotation file")

    raw_references = data.get("references", [])
    raw_samples = data.get("samples", [])
    if not isinstance(raw_references, list) or not isinstance(raw_samples, list):
        raise ValueError(
            "Expected 'references' and 'samples' arrays in annotation file"
        )

    references: dict[str, ThermalReference] = {}
    for entry in raw_references:
        ref_id = str(entry["id"])
        if not ref_id or ref_id in references:
            raise ValueError(f"Invalid or duplicate reference id: {ref_id!r}")
        tiff_file = str(entry["tiff_file"])
        preview_file = str(entry["preview_file"])
        validate_relative_path(tiff_file, f"reference {ref_id} tiff_file")
        validate_relative_path(preview_file, f"reference {ref_id} preview_file")
        polygon = _parse_polygon(entry["polygon"], f"reference {ref_id} polygon")
        references[ref_id] = ThermalReference(
            reference_id=ref_id,
            installation_code=str(entry.get("installation_code", "")),
            tag_id=str(entry.get("tag_id", "")),
            inspection_description=str(entry.get("inspection_description", "")),
            tiff_file=tiff_file,
            preview_file=preview_file,
            width=int(entry["width"]),
            height=int(entry["height"]),
            polygon=polygon,
        )

    samples: dict[int, ThermalSample] = {}
    for entry in raw_samples:
        sample_id = int(entry["id"])
        if sample_id <= 0 or sample_id in samples:
            raise ValueError(f"Invalid or duplicate sample id: {sample_id}")
        file_name = str(entry["file_name"])
        tiff_file = str(entry["tiff_file"])
        preview_file = str(entry["preview_file"])
        validate_relative_path(tiff_file, f"sample {sample_id} tiff_file")
        validate_relative_path(preview_file, f"sample {sample_id} preview_file")
        ref_id = str(entry["reference_id"])
        if ref_id not in references:
            raise ValueError(
                f"Sample {sample_id} references unknown reference_id {ref_id!r}"
            )
        matchable = bool(entry.get("matchable", True))
        raw_gt = entry.get("gt_polygon")
        if matchable:
            gt_polygon: Polygon | None = _parse_polygon(
                raw_gt, f"sample {sample_id} gt_polygon"
            )
        else:
            if raw_gt is not None:
                raise ValueError(
                    f"Unmatchable sample {sample_id} must have gt_polygon=null"
                )
            gt_polygon = None

        samples[sample_id] = ThermalSample(
            sample_id=sample_id,
            file_name=file_name,
            tiff_file=tiff_file,
            preview_file=preview_file,
            width=int(entry["width"]),
            height=int(entry["height"]),
            reference_id=ref_id,
            matchable=matchable,
            gt_polygon=gt_polygon,
        )

    if not samples:
        raise ValueError("Thermal benchmark split contains no samples")
    return references, samples


def validate_relative_path(relative_path: str, label: str) -> None:
    if not relative_path or "\\" in relative_path:
        raise ValueError(f"Invalid relative path for {label}: {relative_path!r}")
    path = PurePosixPath(relative_path)
    if (
        not path.parts
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"Invalid relative path for {label}: {relative_path!r}")


def _parse_polygon(raw_polygon: Any, label: str) -> Polygon:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        raise ValueError(f"Expected at least 3 points for {label}")
    polygon: Polygon = []
    for point in raw_polygon:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = float(point[0]), float(point[1])
        elif isinstance(point, dict) and "x" in point and "y" in point:
            x, y = float(point["x"]), float(point["y"])
        else:
            raise ValueError(f"Invalid polygon vertex in {label}: {point!r}")
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"Non-finite polygon vertex in {label}: {point!r}")
        polygon.append((x, y))
    return polygon


def _verify_dimensions(
    image: NDArray[np.float64],
    expected_width: int,
    expected_height: int,
    label: str,
) -> None:
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"TIFF dimensions ({actual_width}x{actual_height}) do not match metadata "
            f"({expected_width}x{expected_height}) for {label}"
        )
