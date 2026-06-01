import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from sara_thermal_reading.image_alignment.align_two_images_translation_cv2 import (
    align_two_images_translation_cv2,
)
from sara_thermal_reading.image_processing.convert_thermal_to_uint8 import (
    convert_thermal_to_uint8,
)
from sara_thermal_reading.main_thermal_workflow import process_thermal_image

EXAMPLE_DATA = Path(__file__).parent.parent / "example-data"

TRANSLATIONS = [
    (20, 15),
    (-30, 25),
    (50, -40),
    (5, 5),
    (-10, -60),
]

MAX_PIXEL_TOLERANCE = 10


def _translate_image(
    image: NDArray[np.float64], tx: float, ty: float
) -> NDArray[np.float64]:
    """Translate an image by (tx, ty) pixels, filling borders with the image mean."""
    h, w = image.shape[:2]
    matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    mean_value = float(np.mean(image))
    translated: NDArray[np.float64] = cv2.warpAffine(  # type: ignore[assignment]
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(mean_value,),
    )
    return translated


def _max_point_distance(
    polygon_a: list[tuple[int, int]],
    polygon_b: list[tuple[int, int]],
) -> float:
    """Return the maximum Euclidean distance between corresponding polygon points."""
    assert len(polygon_a) == len(polygon_b)
    return max(
        math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(polygon_a, polygon_b)
    )


def _shift_polygon(
    polygon: list[tuple[int, int]], tx: int, ty: int
) -> list[tuple[int, int]]:
    """Shift all polygon points by (tx, ty)."""
    return [(p[0] + tx, p[1] + ty) for p in polygon]


def _load_example(
    directory: str,
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Load a TIFF image and polygon JSON from an example-data subdirectory."""
    image = tifffile.imread(EXAMPLE_DATA / directory / "thermal_image.tiff").astype(
        np.float64
    )
    with open(EXAMPLE_DATA / directory / "polygon.json") as f:
        polygon: list[tuple[int, int]] = [tuple(p) for p in json.load(f)]
    return image, polygon


@pytest.mark.parametrize("tx, ty", TRANSLATIONS)
def test_alignment_asset_example(tx: int, ty: int) -> None:
    reference_image, polygon = _load_example("asset-example")
    source_image = _translate_image(reference_image, tx, ty)

    _temperature, _annotated, warped_polygon, _warped_ref, _score = (
        process_thermal_image(reference_image, source_image, polygon)
    )

    expected_polygon = _shift_polygon(polygon, tx, ty)
    distance = _max_point_distance(expected_polygon, warped_polygon)
    assert distance < MAX_PIXEL_TOLERANCE, (
        f"Max point distance {distance:.1f}px exceeds {MAX_PIXEL_TOLERANCE}px "
        f"for translation ({tx}, {ty})"
    )


@pytest.mark.parametrize("tx, ty", TRANSLATIONS)
def test_alignment_fireplace_example(tx: int, ty: int) -> None:
    reference_image, polygon = _load_example("fireplace-example")
    source_image = _translate_image(reference_image, tx, ty)

    _temperature, _annotated, warped_polygon, _warped_ref, _score = (
        process_thermal_image(reference_image, source_image, polygon)
    )

    expected_polygon = _shift_polygon(polygon, tx, ty)
    distance = _max_point_distance(expected_polygon, warped_polygon)
    assert distance < MAX_PIXEL_TOLERANCE, (
        f"Max point distance {distance:.1f}px exceeds {MAX_PIXEL_TOLERANCE}px "
        f"for translation ({tx}, {ty})"
    )


def test_translation_zero_shift() -> None:
    """Identical images should produce an unchanged polygon and a high alignment score."""
    reference_image, polygon = _load_example("asset-example")
    ref_uint8 = convert_thermal_to_uint8(reference_image)

    warped_polygon, _warped_img, score = align_two_images_translation_cv2(
        ref_uint8, ref_uint8, polygon
    )

    distance = _max_point_distance(polygon, warped_polygon)
    assert distance < 1.0, f"Zero-shift distance should be ~0, got {distance:.1f}px"
    assert score > 0.5, f"Identical images should have high score, got {score:.3f}"


def test_translation_returns_valid_score() -> None:
    """Alignment score should be a float between 0 and 1."""
    reference_image, polygon = _load_example("asset-example")
    source_image = _translate_image(reference_image, 10, 10)
    ref_uint8 = convert_thermal_to_uint8(reference_image)
    src_uint8 = convert_thermal_to_uint8(source_image)

    _warped_polygon, _warped_img, score = align_two_images_translation_cv2(
        ref_uint8, src_uint8, polygon
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_translation_polygon_point_count_preserved() -> None:
    """Output polygon should have the same number of points as the input."""
    reference_image, polygon = _load_example("fireplace-example")
    source_image = _translate_image(reference_image, 20, 15)
    ref_uint8 = convert_thermal_to_uint8(reference_image)
    src_uint8 = convert_thermal_to_uint8(source_image)

    warped_polygon, _warped_img, _score = align_two_images_translation_cv2(
        ref_uint8, src_uint8, polygon
    )

    assert len(warped_polygon) == len(polygon)
