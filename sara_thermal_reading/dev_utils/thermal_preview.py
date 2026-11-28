from io import BytesIO

import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def render_thermal_preview_png(
    thermal_image: NDArray[np.float64],
    colormap_name: str = "RdYlBu_r",
) -> bytes:
    """Render a float64 thermal image to an exact-dimension colormapped PNG."""
    vmin = float(np.nanmin(thermal_image))
    vmax = float(np.nanmax(thermal_image))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        normalized = np.zeros(thermal_image.shape[:2], dtype=np.float64)
    else:
        normalized = np.clip((thermal_image - vmin) / (vmax - vmin), 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0)

    cmap = matplotlib.colormaps[colormap_name]
    rgba = cmap(normalized)
    rgb_uint8 = np.round(rgba[:, :, :3] * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb_uint8, mode="RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
