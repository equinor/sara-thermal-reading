import numpy as np

from sara_thermal_reading.file_io.blob import _np_image_to_jpg_bytes

JPEG_MAGIC = b"\xff\xd8\xff"


def test_rgb_produces_valid_jpeg() -> None:
    """An RGB uint8 array should produce bytes starting with the JPEG magic header."""
    image = np.random.default_rng(42).integers(0, 255, (50, 50, 3), dtype=np.uint8)

    result = _np_image_to_jpg_bytes(image)

    assert result[:3] == JPEG_MAGIC


def test_rgba_converted_to_rgb() -> None:
    """An RGBA input should be converted to RGB and produce a valid JPEG."""
    image = np.random.default_rng(42).integers(0, 255, (50, 50, 4), dtype=np.uint8)

    result = _np_image_to_jpg_bytes(image)

    assert result[:3] == JPEG_MAGIC


def test_output_is_bytes_with_nonzero_length() -> None:
    """Return type should be bytes with non-zero length."""
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    result = _np_image_to_jpg_bytes(image)

    assert isinstance(result, bytes)
    assert len(result) > 0
