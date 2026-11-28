from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any

import cv2
import numpy as np

PROTOCOL = "thermal-reading@v001"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.7

PER_IMAGE_COLUMNS: list[dict[str, str]] = [
    {"key": "matchable", "label": "Matchable", "format": "boolean"},
    {"key": "accepted", "label": "Accepted", "format": "boolean"},
    {"key": "confidence", "label": "Confidence", "format": "percent"},
    {"key": "polygon_iou", "label": "Polygon IoU", "format": "number"},
    {"key": "centroid_error_px", "label": "Centroid Err", "format": "pixels"},
    {"key": "gt_temp_c", "label": "GT Temp", "format": "celsius"},
    {"key": "pred_temp_c", "label": "Pred Temp", "format": "celsius"},
    {"key": "temp_error_c", "label": "Temp Err", "format": "celsius"},
]

Polygon = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class SampleThermalEvaluation:
    matchable: bool
    accepted: bool
    passed: bool
    missed: bool
    false_positive: bool
    confidence: float
    polygon_iou: float | None
    centroid_error_px: float | None
    gt_temp_c: float | None
    pred_temp_c: float | None
    temp_error_c: float | None


@dataclass
class ThermalBenchmarkMetrics:
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    matchable_count: int = 0
    iou_sum: float = 0.0
    centroid_error_sum: float = 0.0
    temp_error_count: int = 0
    temp_error_sum: float = 0.0
    temp_error_max: float | None = None

    def add_sample(
        self,
        matchable: bool,
        width: int,
        height: int,
        gt_polygon: Polygon | None,
        predicted_polygon: Polygon,
        matching_confidence: float,
        gt_temp_c: float | None,
        pred_temp_c: float | None,
    ) -> SampleThermalEvaluation:
        accepted = matching_confidence >= self.confidence_threshold
        polygon_iou: float | None = None
        centroid_error_px: float | None = None
        temp_error_c: float | None = None

        if matchable and gt_polygon is not None:
            polygon_iou = compute_polygon_iou(
                predicted_polygon, gt_polygon, width=width, height=height
            )
            centroid_error_px = compute_polygon_centroid_error(
                predicted_polygon, gt_polygon
            )
            self.matchable_count += 1
            self.iou_sum += polygon_iou
            self.centroid_error_sum += centroid_error_px
            if gt_temp_c is not None and pred_temp_c is not None:
                temp_error_c = abs(pred_temp_c - gt_temp_c)
                self.temp_error_count += 1
                self.temp_error_sum += temp_error_c
                self.temp_error_max = (
                    temp_error_c
                    if self.temp_error_max is None
                    else max(self.temp_error_max, temp_error_c)
                )

        alignment_valid = (
            matchable and polygon_iou is not None and polygon_iou >= self.iou_threshold
        )

        if alignment_valid and accepted:
            self.true_positives += 1
            passed, missed, false_positive = True, False, False
        elif alignment_valid and not accepted:
            self.false_negatives += 1
            passed, missed, false_positive = False, True, False
        elif not alignment_valid and accepted:
            self.false_positives += 1
            passed, missed, false_positive = False, False, True
        else:
            self.true_negatives += 1
            passed, missed, false_positive = True, False, False

        return SampleThermalEvaluation(
            matchable=matchable,
            accepted=accepted,
            passed=passed,
            missed=missed,
            false_positive=false_positive,
            confidence=matching_confidence,
            polygon_iou=polygon_iou,
            centroid_error_px=centroid_error_px,
            gt_temp_c=gt_temp_c,
            pred_temp_c=pred_temp_c,
            temp_error_c=temp_error_c,
        )

    def as_evaluations(self) -> dict[str, dict[str, Any]]:
        return {
            "alignment_classification": {
                "labels": ["good_alignment", "bad_or_unmatchable"],
                "confusion_matrix": [
                    [self.true_positives, self.false_negatives],
                    [self.false_positives, self.true_negatives],
                ],
                "matched_count": self.true_positives,
            }
        }

    def as_summary_metrics(self) -> list[dict[str, Any]]:
        tp = self.true_positives
        fn = self.false_negatives
        fp = self.false_positives
        tn = self.true_negatives

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = compute_f1(precision, recall)
        fp_rate = safe_ratio(fp, fp + tn)
        mean_iou = safe_ratio(self.iou_sum, self.matchable_count)
        mean_centroid_err = safe_ratio(self.centroid_error_sum, self.matchable_count)
        temp_mae = safe_ratio(self.temp_error_sum, self.temp_error_count)

        return [
            {
                "key": "precision",
                "label": "Precision",
                "value": precision,
                "format": "percent",
                "subtitle": f"TP {tp} / FP {fp}",
                "primary": True,
            },
            {
                "key": "recall",
                "label": "Recall",
                "value": recall,
                "format": "percent",
                "subtitle": f"TP {tp} / FN {fn}",
                "primary": True,
            },
            {
                "key": "f1",
                "label": "F1",
                "value": f1,
                "format": "percent",
                "primary": True,
            },
            {
                "key": "mean_iou",
                "label": "Mean IoU",
                "value": mean_iou,
                "format": "number",
                "subtitle": f"Pass >= {self.iou_threshold:.2f}",
                "primary": True,
            },
            {
                "key": "temp_mae_c",
                "label": "Temp MAE",
                "value": temp_mae,
                "format": "celsius",
                "subtitle": f"{self.temp_error_count} samples",
                "primary": True,
            },
            {
                "key": "fp_rate",
                "label": "FP Rate",
                "value": fp_rate,
                "format": "percent",
                "subtitle": f"FP {fp} / TN {tn}",
                "primary": False,
            },
            {
                "key": "temp_max_err_c",
                "label": "Temp Max Err",
                "value": self.temp_error_max,
                "format": "celsius",
                "primary": False,
            },
            {
                "key": "mean_centroid_error_px",
                "label": "Mean Centroid Err",
                "value": mean_centroid_err,
                "format": "pixels",
                "primary": False,
            },
        ]


def compute_polygon_iou(
    polygon_a: Polygon,
    polygon_b: Polygon,
    width: int,
    height: int,
) -> float:
    mask_a = rasterize_polygon(polygon_a, width=width, height=height)
    mask_b = rasterize_polygon(polygon_b, width=width, height=height)
    union = int(np.count_nonzero(mask_a | mask_b))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    return float(intersection) / float(union)


def rasterize_polygon(
    polygon: Polygon,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], (1,))
    return mask.astype(bool)


def compute_polygon_centroid_error(polygon_a: Polygon, polygon_b: Polygon) -> float:
    ax, ay = _polygon_centroid(polygon_a)
    bx, by = _polygon_centroid(polygon_b)
    return hypot(ax - bx, ay - by)


def _polygon_centroid(polygon: Polygon) -> tuple[float, float]:
    pts = np.asarray(polygon, dtype=np.float64)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def compute_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
