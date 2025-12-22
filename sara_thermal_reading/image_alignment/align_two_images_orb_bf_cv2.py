from typing import cast

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray


def align_two_images_orb_bf_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Align source image to reference image and transform the polygon accordingly.

    Args:
        reference_image: The reference image (target alignment)
        source_image: The source image to be aligned
        roi_polygon: Polygon points defined in reference image coordinates

    Returns:
        tuple: (warped_polygon, aligned_source_image)
    """
    # Define the polygon points from reference image
    polygon_points = np.array(roi_polygon, dtype=np.float32)

    # Convert images to grayscale
    gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_reference, None)
    keypoints_src, descriptors_src = orb.detectAndCompute(gray_source, None)

    if descriptors_ref is None or descriptors_src is None:
        logger.error("Could not find features in one or both images")
        # Return original source image with original polygon
        return polygon_points.reshape(-1, 1, 2), source_image

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_ref, descriptors_src)

    if len(matches) < 4:
        logger.error(f"Insufficient matches found: {len(matches)}. Need at least 4.")
        return polygon_points.reshape(-1, 1, 2), source_image

    # Sort them in order of distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take only the best matches (top 50% or max 100)
    num_good_matches = min(len(matches), max(len(matches) // 2, 100))
    good_matches = matches[:num_good_matches]

    logger.info(
        f"Using {num_good_matches} good matches out of {len(matches)} total matches"
    )

    # Extract location of good matches
    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_src = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_src[i, :] = keypoints_src[match.trainIdx].pt

    # Calculate homography to transform source image TO reference image coordinates
    H, mask = cv2.findHomography(
        points_src, points_ref, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
    )

    # Check if homography is valid
    if H is None:
        logger.error("Could not compute homography")
        return polygon_points.reshape(-1, 1, 2), source_image

    num_inliers = np.sum(mask) if mask is not None else 0
    logger.info(
        f"Homography computed with {num_inliers} inliers out of {len(good_matches)} matches"
    )

    if num_inliers < 10:
        logger.warning(f"Poor homography quality - only {num_inliers} inliers")
        # Could add fallback behavior here

    # Check if homography is reasonable (not too distorted)
    try:
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 0.1 or abs(det) > 10:
            logger.warning(f"Homography appears distorted (determinant: {det:.3f})")
    except:
        logger.warning("Could not validate homography determinant")

    # Warp source image to align with reference image
    aligned_image = cv2.warpPerspective(
        source_image, H, (reference_image.shape[1], reference_image.shape[0])
    )

    # Since we're aligning source TO reference, the polygon coordinates
    # from the reference image remain valid for the aligned result
    warped_polygon = polygon_points.reshape(-1, 1, 2)

    return cast(NDArray[np.float32], warped_polygon), cast(
        NDArray[np.uint8], aligned_image
    )
