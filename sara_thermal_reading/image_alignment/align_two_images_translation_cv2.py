import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.config import settings

logger = logging.getLogger(__name__)


def align_two_images_translation_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], NDArray[np.uint8], float]:
    """
    Align reference image to source image using translation only, and
    transform the polygon accordingly.

    Uses phase correlation to estimate the (dx, dy) shift between images.
    This is appropriate when the camera moves only slightly between captures
    and no rotation or perspective change is expected.

    Args:
        reference_image: The reference image (source of polygon)
        source_image: The source image (target alignment)
        roi_polygon: Polygon points defined in reference image coordinates

    Returns:
        tuple: (translated_polygon, translated_reference_image)
        translated_polygon: Polygon shifted to source image coordinates
        translated_reference_image: Reference image shifted to match source image
        alignment_score: Phase correlation response (confidence of the match)
    """
    # Phase correlation requires float32 images
    reference_float = reference_image.astype(np.float32)
    source_float = source_image.astype(np.float32)

    # The Hann window reduces edge effects by weighting the center of the image the most.
    hann = cv2.createHanningWindow(reference_image.shape, cv2.CV_32F).T
    # Estimate translation: phaseCorrelate returns (dx, dy) such that
    # the source image is shifted by (dx, dy) relative to the reference.
    (dx, dy), phase_correlation = cv2.phaseCorrelate(
        reference_float, source_float, hann
    )

    logger.info(
        f"Estimated translation: dx={dx:.2f}, dy={dy:.2f}, response={phase_correlation:.3f}"
    )

    # Apply the translation to the reference image to align it with the source
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    translated_reference_image: NDArray[np.uint8] = cv2.warpAffine(
        reference_image,
        translation_matrix,
        (source_image.shape[1], source_image.shape[0]),
    ).astype(np.uint8)

    # Shift the polygon by the same (dx, dy) to move it into source image coordinates
    translated_polygon: list[tuple[int, int]] = [
        (int(round(x + dx)), int(round(y + dy))) for x, y in roi_polygon
    ]

    return translated_polygon, translated_reference_image, phase_correlation
