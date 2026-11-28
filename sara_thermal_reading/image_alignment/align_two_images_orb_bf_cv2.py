import logging
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MIN_MATCHES_FOR_HOMOGRAPHY = 4
MIN_CONFIDENT_INLIERS = 10
MIN_HOMOGRAPHY_DET = 0.1
MAX_HOMOGRAPHY_DET = 10.0


def align_two_images_orb_bf_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: Sequence[tuple[float, float]],
) -> tuple[list[tuple[int, int]], NDArray[np.uint8], float]:
    """
    Align reference image to source image using ORB keypoints and RANSAC homography,
    returning the warped polygon, warped reference image, and a 0..1 confidence score.
    """
    fallback_polygon = [(int(round(x)), int(round(y))) for x, y in roi_polygon]
    polygon_points = np.array(roi_polygon, dtype=np.float32)

    gray_reference = _to_grayscale(reference_image)
    gray_source = _to_grayscale(source_image)

    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_reference, None)
    keypoints_src, descriptors_src = orb.detectAndCompute(gray_source, None)

    if descriptors_ref is None or descriptors_src is None:
        logger.error("Could not find features in one or both images")
        return fallback_polygon, reference_image, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_src)

    if len(matches) < MIN_MATCHES_FOR_HOMOGRAPHY:
        logger.error(
            f"Insufficient matches found: {len(matches)}. Need at least {MIN_MATCHES_FOR_HOMOGRAPHY}."
        )
        return fallback_polygon, reference_image, 0.0

    matches = sorted(matches, key=lambda match: match.distance)
    num_good_matches = min(len(matches), max(len(matches) // 2, 100))
    good_matches = matches[:num_good_matches]

    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_src = np.zeros((len(good_matches), 2), dtype=np.float32)
    for idx, match in enumerate(good_matches):
        points_ref[idx, :] = keypoints_ref[match.queryIdx].pt
        points_src[idx, :] = keypoints_src[match.trainIdx].pt

    homography, mask = cv2.findHomography(
        points_ref, points_src, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
    )
    if homography is None:
        logger.error("Could not compute homography")
        return fallback_polygon, reference_image, 0.0

    num_inliers = int(np.sum(mask)) if mask is not None else 0
    confidence = _compute_homography_confidence(
        homography, num_inliers, len(good_matches)
    )

    aligned_reference_image = cv2.warpPerspective(
        reference_image, homography, (source_image.shape[1], source_image.shape[0])
    )
    warped_polygon_array = cv2.perspectiveTransform(
        polygon_points.reshape(-1, 1, 2), homography
    )
    warped_polygon_list = [
        (int(round(float(pt[0]))), int(round(float(pt[1]))))
        for pt in warped_polygon_array.reshape(-1, 2)
    ]

    return (
        warped_polygon_list,
        cast(NDArray[np.uint8], aligned_reference_image),
        confidence,
    )


def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if len(image.shape) == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return image


def _compute_homography_confidence(
    homography: np.ndarray,
    num_inliers: int,
    num_good_matches: int,
) -> float:
    if num_good_matches <= 0 or num_inliers <= 0:
        return 0.0
    try:
        det = float(np.linalg.det(cast(NDArray[np.float64], homography[:2, :2])))
    except np.linalg.LinAlgError:
        return 0.0
    if abs(det) < MIN_HOMOGRAPHY_DET or abs(det) > MAX_HOMOGRAPHY_DET:
        logger.warning(f"Homography appears distorted (determinant: {det:.3f})")
        return 0.0

    inlier_ratio = float(num_inliers) / float(num_good_matches)
    inlier_support = min(1.0, float(num_inliers) / float(MIN_CONFIDENT_INLIERS))
    return float(min(1.0, max(0.0, inlier_ratio * inlier_support)))
