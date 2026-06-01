import numpy as np

from sara_thermal_reading.image_processing.convert_thermal_to_uint8 import (
    calculate_clip_values_from_percentiles,
    convert_thermal_to_uint8,
)


def test_convert_output_dtype_and_range() -> None:
    """Output should be uint8 with values spanning 0 to 255."""
    gradient = np.linspace(10.0, 40.0, 100 * 100, dtype=np.float64).reshape(100, 100)

    result = convert_thermal_to_uint8(gradient)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_convert_with_explicit_clip_range() -> None:
    """Values outside clip range should be clamped; endpoints map to 0 and 255."""
    image = np.linspace(0.0, 100.0, 200 * 200, dtype=np.float64).reshape(200, 200)

    result = convert_thermal_to_uint8(image, clip_range=(20.0, 80.0))

    assert result.dtype == np.uint8
    # Pixels at exactly vmin should map to 0, at vmax to 255
    assert result[0, 0] == 0  # 0.0 clipped to 20.0 -> maps to 0
    assert result[-1, -1] == 255  # 100.0 clipped to 80.0 -> maps to 255


def test_convert_no_clip_range_uses_image_minmax() -> None:
    """Without clip_range, image min maps to 0 and image max maps to 255."""
    image = np.array([[5.0, 15.0], [25.0, 50.0]], dtype=np.float64)

    result = convert_thermal_to_uint8(image)

    assert result.dtype == np.uint8
    # The minimum value (5.0) should map to 0
    assert result[0, 0] == 0
    # The maximum value (50.0) should map to 255
    assert result[1, 1] == 255


def test_convert_uniform_image() -> None:
    """Uniform image (vmin == vmax) should not crash."""
    image = np.full((50, 50), 20.0, dtype=np.float64)

    result = convert_thermal_to_uint8(image)

    assert result.dtype == np.uint8
    assert result.shape == (50, 50)


def test_clip_percentiles_exclude_outliers() -> None:
    """A single extreme outlier should be excluded by default percentile clipping."""
    image = np.full((200, 200), 20.0, dtype=np.float64)
    image[0, 0] = 1000.0  # extreme outlier

    vmin, vmax = calculate_clip_values_from_percentiles(image)

    # vmax should be close to 20.0, not pulled up to 1000.0
    assert vmax < 100.0
    assert vmin >= 20.0


def test_clip_percentiles_0_100_equals_minmax() -> None:
    """Percentiles 0.0 and 100.0 should return the actual min and max."""
    rng = np.random.default_rng(42)
    image = rng.uniform(10.0, 50.0, (100, 100)).astype(np.float64)

    vmin, vmax = calculate_clip_values_from_percentiles(
        image, clip_percentile_min=0.0, clip_percentile_max=100.0
    )

    assert vmin == float(np.min(image))
    assert vmax == float(np.max(image))


def test_clip_percentiles_returns_ordered_floats() -> None:
    """Return values should be floats with vmin <= vmax."""
    rng = np.random.default_rng(42)
    image = rng.normal(25.0, 5.0, (200, 200)).astype(np.float64)

    vmin, vmax = calculate_clip_values_from_percentiles(image)

    assert isinstance(vmin, float)
    assert isinstance(vmax, float)
    assert vmin <= vmax
