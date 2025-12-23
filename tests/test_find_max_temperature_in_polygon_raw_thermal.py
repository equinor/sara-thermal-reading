import numpy as np

from sara_thermal_reading.image_processing.find_max_temperature_in_polygon_raw_thermal import (
    find_max_temperature_in_polygon_raw_thermal,
)


def test_find_max_temperature_in_polygon_raw_thermal() -> None:

    thermal_image = np.full((100, 100), 20.0, dtype=np.float64)

    # Define a polygon square
    polygon_points = np.array(
        [[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32
    )

    # Set a high temperature point inside the polygon
    expected_max_temp = 45.5
    expected_loc = (50, 50)
    thermal_image[expected_loc[1], expected_loc[0]] = expected_max_temp

    # Set a higher temperature point outside the polygon
    thermal_image[10, 10] = 100.0

    max_temp, max_loc = find_max_temperature_in_polygon_raw_thermal(
        thermal_image, polygon_points
    )

    assert max_temp == expected_max_temp
    assert max_loc == expected_loc
    assert isinstance(max_temp, float)
    assert isinstance(max_loc, tuple)
    assert len(max_loc) == 2
    assert isinstance(max_loc[0], int)
    assert isinstance(max_loc[1], int)
