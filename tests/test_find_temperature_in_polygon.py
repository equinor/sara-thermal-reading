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
