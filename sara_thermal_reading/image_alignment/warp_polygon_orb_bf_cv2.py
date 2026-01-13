import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.image_alignment.base import ImageAlignmentStrategy

logger = logging.getLogger(__name__)


class WarpPolygonAligner(ImageAlignmentStrategy):
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
        # Define the polygon points in Image1 (x, y)
        polygon_points = np.array(roi_polygon, dtype=np.float32)

        # Convert images to grayscale
        if len(reference_image.shape) == 3:
            gray1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = reference_image

        if len(source_image.shape) == 3:
            gray2 = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = source_image

        # Detect ORB keypoints and compute descriptors
        orb = cv2.ORB_create()  # type: ignore
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        if descriptors1 is None or descriptors2 is None:
            logger.error("Could not find features in one or both images")
            return roi_polygon

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort them in order of distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        if len(matches) < 4:
            logger.error(
                f"Insufficient matches found: {len(matches)}. Need at least 4."
            )
            return roi_polygon

        # Calculate the homography matrix
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        if H is None:
            logger.error("Could not compute homography")
            return roi_polygon

        # Warp the polygon points from Image1 to Image2
        warped_polygon = cv2.perspectiveTransform(polygon_points.reshape(-1, 1, 2), H)

        # Convert warped polygon array to list of tuples
        warped_polygon_list = [
            (int(pt[0]), int(pt[1])) for pt in warped_polygon.reshape(-1, 2)
        ]

        return warped_polygon_list
