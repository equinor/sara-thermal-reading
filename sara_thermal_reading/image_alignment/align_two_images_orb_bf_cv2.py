import logging
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.image_alignment.base import ImageAlignmentStrategy

logger = logging.getLogger(__name__)


class OrbBfCv2Aligner(ImageAlignmentStrategy):
    def align(
        self,
        reference_image: NDArray[np.uint8],
        source_image: NDArray[np.uint8],
        roi_polygon: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Aligns the reference image context to the source image and warps the polygon.

        Args:
            reference_image: The reference thermal image as a numpy array (uint8).
            source_image: The source (current) thermal image as a numpy array (uint8).
            roi_polygon: A list of (x, y) tuples representing the polygon on the
                reference image.

        Returns:
            list[tuple[int, int]]: The coordinates of the polygon warped to match
                the source image.
        """
        # Define the polygon points from reference image
        polygon_points = np.array(roi_polygon, dtype=np.float32)

        # Convert images to grayscale if they are not already
        if len(reference_image.shape) == 3:
            gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_reference = reference_image

        if len(source_image.shape) == 3:
            gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_source = source_image

        # Detect ORB keypoints and compute descriptors
        orb = cv2.ORB_create(nfeatures=1000)  # type: ignore
        keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_reference, None)
        keypoints_src, descriptors_src = orb.detectAndCompute(gray_source, None)

        if descriptors_ref is None or descriptors_src is None:
            logger.error("Could not find features in one or both images")
            # Return original polygon (fallback)
            return roi_polygon

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors_ref, descriptors_src)

        if len(matches) < 4:
            logger.error(
                f"Insufficient matches found: {len(matches)}. Need at least 4."
            )
            return roi_polygon

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
            points_ref,
            points_src,
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
        )

        # Check if homography is valid
        if H is None:
            logger.error("Could not compute homography")
            return roi_polygon

        num_inliers = np.sum(mask) if mask is not None else 0
        logger.info(
            f"Homography computed with {num_inliers} inliers out of {len(good_matches)} matches"
        )

        if num_inliers < 10:
            logger.warning(f"Poor homography quality - only {num_inliers} inliers")
            # Could add fallback behavior here

        # Check if homography is reasonable (not too distorted)
        try:
            det = np.linalg.det(cast(NDArray[np.float64], H[:2, :2]))
            if abs(det) < 0.1 or abs(det) > 10:
                logger.warning(f"Homography appears distorted (determinant: {det:.3f})")
        except:
            logger.warning("Could not validate homography determinant")

        # Warp the polygon points from Reference to Source
        warped_polygon_array = cv2.perspectiveTransform(
            polygon_points.reshape(-1, 1, 2), H
        )

        # Convert warped polygon array to list of tuples
        warped_polygon_list = [
            (int(pt[0]), int(pt[1])) for pt in warped_polygon_array.reshape(-1, 2)
        ]

        return warped_polygon_list


def align_two_images_orb_bf_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], NDArray[np.uint8]]:
    """
    Deprecated: Use OrbBfCv2Aligner class instead.
    """
    aligner = OrbBfCv2Aligner()
    warped_poly = aligner.align(reference_image, source_image, roi_polygon)
    return warped_poly, reference_image
