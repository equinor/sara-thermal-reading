import cv2
import numpy as np
from numpy.typing import NDArray


def find_max_temperature_in_polygon_raw_thermal(
    thermal_image: NDArray[np.float64], polygon_points: NDArray[np.float32]
) -> tuple[float, tuple[int, int]]:
    """
    Finds the maximum temperature within a polygon region in a raw thermal image.

    Args:
        thermal_image: The raw thermal image (float64) with temperature values.
        polygon_points: The polygon points (float32) defining the region of interest.

    Returns:
        A tuple containing:
        - max_temperature: The maximum temperature value found within the polygon.
        - max_temp_location: The (x, y) coordinates of the maximum temperature.
    """
    # Create a mask for the polygon
    mask = np.zeros(thermal_image.shape, dtype=np.uint8)

    # Convert polygon points to integer for fillPoly
    int_polygon_points = polygon_points.astype(np.int32)

    cv2.fillPoly(mask, [int_polygon_points], (255,))

    # Find the location of the maximum value within the masked region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thermal_image, mask=mask)

    return float(max_val), (int(max_loc[0]), int(max_loc[1]))
