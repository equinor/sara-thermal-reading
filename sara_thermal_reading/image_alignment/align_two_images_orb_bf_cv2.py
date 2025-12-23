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
    Align reference image to source image and transform the polygon accordingly.

    Args:
        reference_image: The reference image (source of polygon)
        source_image: The source image (target alignment)
        roi_polygon: Polygon points defined in reference image coordinates

    Returns:
        tuple: (warped_polygon, aligned_reference_image)
        warped_polygon: Polygon transformed to source image coordinates
        aligned_reference_image: Reference image warped to match source image geometry
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
        # Return original polygon and reference image (fallback)
        return polygon_points.reshape(-1, 1, 2), reference_image

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_ref, descriptors_src)

    if len(matches) < 4:
        logger.error(f"Insufficient matches found: {len(matches)}. Need at least 4.")
        return polygon_points.reshape(-1, 1, 2), reference_image

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

    # Calculate homography to transform REFERENCE image TO SOURCE image coordinates
    # Note: points_ref -> points_src
    H, mask = cv2.findHomography(
        points_ref, points_src, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
    )

    # Check if homography is valid
    if H is None:
        logger.error("Could not compute homography")
        return polygon_points.reshape(-1, 1, 2), reference_image

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

    # Warp reference image to align with source image
    aligned_reference_image = cv2.warpPerspective(
        reference_image, H, (source_image.shape[1], source_image.shape[0])
    )

    # Warp the polygon points from Reference to Source
    warped_polygon = cv2.perspectiveTransform(polygon_points.reshape(-1, 1, 2), H)

    return cast(NDArray[np.float32], warped_polygon), cast(
        NDArray[np.uint8], aligned_reference_image
    )
