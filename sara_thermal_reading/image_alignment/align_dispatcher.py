from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)
from sara_thermal_reading.image_alignment.align_two_images_translation_cv2 import (
    align_two_images_translation_cv2,
)

AlignmentMethod = Literal["phase_correlation", "orb_bf"]
SUPPORTED_ALIGNMENT_METHODS: tuple[AlignmentMethod, ...] = (
    "phase_correlation",
    "orb_bf",
)


@dataclass(frozen=True)
class AlignmentResult:
    warped_polygon: list[tuple[int, int]]
    warped_reference_image: NDArray[np.uint8]
    raw_score: float
    matching_confidence: float


def phase_correlation_to_matching_confidence(phase_correlation: float) -> float:
    """
    Convert phase correlation response to a confidence score between 0 and 1.
    """
    if phase_correlation < settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION:
        return 0.0
    if phase_correlation > settings.CONFIDENCE_CALC_LINEAR_MAX_PHASE_CORRELATION:
        return 1.0
    return (
        phase_correlation - settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION
    ) / (
        settings.CONFIDENCE_CALC_LINEAR_MAX_PHASE_CORRELATION
        - settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION
    )


def align_thermal_images(
    reference_image_uint8: NDArray[np.uint8],
    source_image_uint8: NDArray[np.uint8],
    roi_polygon: Sequence[tuple[float, float]],
    method: AlignmentMethod = "phase_correlation",
) -> AlignmentResult:
    if method == "phase_correlation":
        warped_polygon, warped_ref, raw_score = align_two_images_translation_cv2(
            reference_image_uint8, source_image_uint8, roi_polygon
        )
        return AlignmentResult(
            warped_polygon=warped_polygon,
            warped_reference_image=warped_ref,
            raw_score=float(raw_score),
            matching_confidence=phase_correlation_to_matching_confidence(
                float(raw_score)
            ),
        )
    if method == "orb_bf":
        warped_polygon, warped_ref, confidence = align_two_images_orb_bf_cv2(
            reference_image_uint8, source_image_uint8, roi_polygon
        )
        return AlignmentResult(
            warped_polygon=warped_polygon,
            warped_reference_image=warped_ref,
            raw_score=float(confidence),
            matching_confidence=float(confidence),
        )
    raise ValueError(f"Unsupported alignment method: {method}")
