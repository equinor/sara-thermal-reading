import logging
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from mpl_point_clicker import clicker

logger = logging.getLogger(__name__)


def create_reference_polygon(
    ref_image_fff: np.ndarray,
    ref_image_jpg: np.ndarray | None,
    ref_polygon: list[tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:

    fig = plt.figure()

    if ref_image_jpg is None:
        plt.imshow(ref_image_fff, cmap="jet")
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(ref_image_jpg, cmap="jet")
        plt.subplot(1, 2, 2)
        plt.imshow(ref_image_fff, cmap="jet")

    ax = plt.gca()

    polygon_points: np.ndarray
    if ref_polygon is None:
        polygon_points = np.empty((0, 2), dtype=int)
    else:
        polygon_points = np.array(ref_polygon)
    patch = plt.Polygon(
        polygon_points, closed=True, fill=None, edgecolor="r", linewidth=1
    )
    ax.add_patch(patch)

    klicker = clicker(ax, classes=["image"], markers=["x"], colors=["red"])
    klicker.set_positions({"image": polygon_points})
    klicker._update_points()

    def on_click(event: Any) -> None:
        points = klicker.get_positions()["image"]
        patch.set_xy(points)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

    points = klicker.get_positions().get("image", [])
    if len(points) < 3:
        return None
    points_arr = np.asarray(points)
    polygon_list: list[tuple[int, int]] = [
        tuple(point) for point in points_arr.astype(int).tolist()
    ]
    return polygon_list
