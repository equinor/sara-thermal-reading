import numpy as np
import pytest
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import ImageAlignmentMethod
from sara_thermal_reading.dev_utils.synthetic_data import (
    generate_synthetic_alignment_data,
)
from sara_thermal_reading.image_alignment import get_alignment_strategy


@pytest.mark.parametrize("method", list(ImageAlignmentMethod))
@pytest.mark.parametrize("seed", range(11))  # for deterministic testing
def test_alignment_strategy_execution(
    method: ImageAlignmentMethod,
    seed: int,
) -> None:
    ref_img, src_img, polygon, (tx, ty) = generate_synthetic_alignment_data(seed)

    strategy = get_alignment_strategy(method)

    warped_polygon = strategy.align(ref_img, src_img, polygon)

    assert isinstance(warped_polygon, list)
    assert len(warped_polygon) == len(polygon)
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in warped_polygon)
    assert all(isinstance(c, int) for pt in warped_polygon for c in pt)

    tolerance = 10.0  # pixels

    for (orig_x, orig_y), (warp_x, warp_y) in zip(polygon, warped_polygon):
        expected_x = orig_x + tx
        expected_y = orig_y + ty

        distance = np.sqrt((warp_x - expected_x) ** 2 + (warp_y - expected_y) ** 2)

        assert distance < tolerance, (
            f"Point diverged too much: Original({orig_x}, {orig_y}) -> Warped({warp_x}, {warp_y}). "
            f"Expected displacement ({tx:.2f}, {ty:.2f}), Actual dist {distance:.2f}"
        )
