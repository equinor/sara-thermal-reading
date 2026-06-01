from pathlib import Path

import numpy as np

from sara_thermal_reading.file_io.tiff_loader import (
    load_thermal_tiff,
    load_thermal_tiff_from_bytes,
)

EXAMPLE_DATA = Path(__file__).parent.parent / "example-data"


def test_load_thermal_tiff_shape_and_dtype() -> None:
    """Loading a TIFF should return a 2D float numpy array."""
    path = str(EXAMPLE_DATA / "asset-example" / "thermal_image.tiff")

    image = load_thermal_tiff(path)

    assert isinstance(image, np.ndarray)
    assert image.ndim == 2
    assert np.issubdtype(image.dtype, np.floating)


def test_load_thermal_tiff_from_bytes_matches_file() -> None:
    """Bytes-based loader should produce an array identical to file-based loader."""
    path = EXAMPLE_DATA / "asset-example" / "thermal_image.tiff"

    image_from_file = load_thermal_tiff(str(path))
    image_from_bytes = load_thermal_tiff_from_bytes(path.read_bytes())

    np.testing.assert_array_equal(image_from_file, image_from_bytes)


def test_load_thermal_tiff_fireplace() -> None:
    """Fireplace TIFF should load with expected shape (540, 960) and float dtype."""
    path = str(EXAMPLE_DATA / "fireplace-example" / "thermal_image.tiff")

    image = load_thermal_tiff(path)

    assert isinstance(image, np.ndarray)
    assert image.shape == (540, 960)
    assert np.issubdtype(image.dtype, np.floating)
