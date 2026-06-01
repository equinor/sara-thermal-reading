import numpy as np

from sara_thermal_reading.image_processing.find_temperature_in_polygon import (
    find_temperature_in_polygon,
)


def test_find_temperature_in_polygon_max() -> None:
    thermal_image = np.full((100, 100), 20.0, dtype=np.float64)
    polygon_points = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)
    thermal_image[40:70, 40:70] = 35.0
    # Set a high temperature point inside the polygon
    thermal_image[35, 35] = 100.0

    max_expected = 100.0

    max_temp = find_temperature_in_polygon(
        thermal_image, polygon_points, percentile=100.0
    )

    assert max_temp == max_expected


def test_find_temperature_in_polygon_percentile() -> None:
    thermal_image = np.full((100, 100), 20.0, dtype=np.float64)
    polygon_points = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)
    thermal_image[40:70, 40:70] = 35.0
    # Set a high temperature point inside the polygon
    thermal_image[35, 35] = 100.0

    temp = find_temperature_in_polygon(thermal_image, polygon_points, percentile=95.0)

    # The 95th percentile should be below the single outlier of 100.0
    assert temp < 100.0
    # Most of the polygon area is 20.0 or 35.0, so P95 should be around 35.0
    assert temp == 35.0


def test_find_temperature_min_percentile() -> None:
    """percentile=0.0 should return the minimum value inside the polygon."""
    thermal_image = np.full((100, 100), 20.0, dtype=np.float64)
    polygon_points = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.int32)
    # Place a cold spot inside the polygon
    thermal_image[50, 50] = 5.0

    temp = find_temperature_in_polygon(thermal_image, polygon_points, percentile=0.0)

    assert temp == 5.0


def test_find_temperature_median() -> None:
    """percentile=50.0 should return the median value inside the polygon."""
    thermal_image = np.full((100, 100), 10.0, dtype=np.float64)
    polygon_points = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
    # Set the upper half of the polygon region to 30.0
    thermal_image[20:50, 20:80] = 30.0

    temp = find_temperature_in_polygon(thermal_image, polygon_points, percentile=50.0)

    # Roughly half at 10.0 and half at 30.0 -> median should be near one of the two
    assert 10.0 <= temp <= 30.0


def test_find_temperature_triangle_polygon() -> None:
    """Should work with a triangular (3-point) polygon."""
    thermal_image = np.full((100, 100), 15.0, dtype=np.float64)
    # Triangle covering part of the image
    triangle = np.array([[50, 10], [90, 90], [10, 90]], dtype=np.int32)
    # Set a hot region inside the triangle
    thermal_image[60:80, 30:70] = 45.0

    temp = find_temperature_in_polygon(thermal_image, triangle, percentile=100.0)

    assert temp == 45.0
