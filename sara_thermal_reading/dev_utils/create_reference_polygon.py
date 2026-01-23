import json
import logging
import os
from pathlib import Path
from typing import Any, List, Sequence, cast

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_point_clicker import clicker
from numpy.typing import NDArray

from sara_thermal_reading.file_io.blob import (
    BlobStorageLocation,
    download_blob_to_image,
    upload_bytes_to_blob,
)
from sara_thermal_reading.file_io.fff_loader import load_fff
from sara_thermal_reading.file_io.file_utils import (
    download_fff_image,
    download_reference_polygon,
)

logger = logging.getLogger(__name__)


def create_reference_polygon(
    fff_image_path: Path, polygon_json_output_path: Path
) -> None:
    """
    Opens an interactive window to draw a polygon on the thermal image.
    Saves the polygon coordinates to a JSON file.
    """
    if not fff_image_path.exists():
        logger.error(f"Image file not found: {fff_image_path}")
        return

    thermal_image = load_fff(str(fff_image_path))

    norm_image = np.zeros_like(thermal_image)
    cv2.normalize(thermal_image, norm_image, 0, 255, cv2.NORM_MINMAX)
    display_image_gray = norm_image.astype(np.uint8)

    base_image = cv2.applyColorMap(display_image_gray, cv2.COLORMAP_JET)
    display_image = base_image.copy()

    points: List[List[int]] = []
    window_name = "Draw Polygon (Click to add points, Enter to save)"

    def draw_dotted_line(
        img: np.ndarray,
        pt1: Sequence[int],
        pt2: Sequence[int],
        color: tuple[int, int, int],
        thickness: int = 1,
        gap: int = 10,
    ) -> None:
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0:
            return
        pts = np.linspace(pt1, pt2, int(dist / gap) + 1)
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(
                    img,
                    tuple(pts[i].astype(int)),
                    tuple(pts[i + 1].astype(int)),
                    color,
                    thickness,
                )

    def redraw() -> None:
        nonlocal display_image
        display_image = base_image.copy()
        for i, point in enumerate(points):
            cv2.circle(display_image, tuple(point), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(
                    display_image, tuple(points[i - 1]), tuple(point), (0, 255, 0), 2
                )

        if len(points) > 2:
            draw_dotted_line(
                display_image,
                tuple(points[-1]),
                tuple(points[0]),
                (0, 255, 0),
                thickness=1,
                gap=5,
            )

        cv2.imshow(window_name, display_image)

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if len(points) < 3:
                logger.warning("Polygon must have at least 3 points.")
                continue

            # Close the polygon visually
            cv2.line(display_image, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow(window_name, display_image)
            cv2.waitKey(500)  # Show closed polygon briefly
            break
        elif key == 27:  # Esc
            logger.info("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    with open(polygon_json_output_path, "w") as f:
        json.dump(points, f, indent=4)

    logger.info(f"Polygon saved to {polygon_json_output_path}")
    print(json.dumps(points, indent=4))


def show_cloud_reference_polygon(
    storage_account: str,
    tag: str,
    desc: str,
) -> None:

    base_path = f"{tag}_{desc}"

    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=cast(str, os.getenv("POLYGON_REFERENCE_STORAGE_CONNTECTION_STRING"))
    )

    blob_location = BlobStorageLocation(
        blobContainer="kaa",
        blobName=f"{base_path}/reference_image.fff",
    )
    thermal_image_array = download_fff_image(
        blob_service_client=blob_service_client, blob_storage_location=blob_location
    )

    plot_images_from_blob(
        base_path, blob_service_client, blob_location, thermal_image_array
    )

    plot_polygon(base_path, blob_service_client, blob_location)


def plot_polygon(
    base_path: str,
    blob_service_client: BlobServiceClient,
    blob_location: BlobStorageLocation,
) -> None:
    polygon_json_str: Any = None
    try:
        blob_location.blob_name = f"{base_path}/reference_polygon.json"
        polygon_json_str = download_reference_polygon(
            blob_service_client=blob_service_client, blob_storage_location=blob_location
        )
    except Exception as e:
        logger.warning(f"Failed to load existing polygon JSON: {e}.")

    if (polygon_json_str is not None) and (len(polygon_json_str) > 0):
        x_coordinates = [point[0] for point in polygon_json_str]
        y_coordinates = [point[1] for point in polygon_json_str]
        plt.fill(x_coordinates, y_coordinates, edgecolor="r", fill=False, linewidth=1)
        plt.show()


def plot_images_from_blob(
    base_path: str,
    blob_service_client: BlobServiceClient,
    blob_location: BlobStorageLocation,
    thermal_image_array: NDArray[np.float64] | NDArray[np.uint8],
) -> None:
    try:
        blob_location.blob_name = f"{base_path}/reference_image.jpeg"
        jpeg_array = download_blob_to_image(
            blob_service_client=blob_service_client, blob_storage_location=blob_location
        )

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(jpeg_array, cmap="jet")
        plt.subplot(1, 2, 2)
        plt.imshow(thermal_image_array, cmap="jet")
        plt.subplot(1, 2, 2)

    except Exception as e:
        logger.warning(f"Failed to load JPEG image: {e}. Using FFF image only.")

        plt.figure()
        plt.imshow(thermal_image_array, cmap="jet")
        plt.gca()


def create_cloud_reference_polygon(
    storage_account: str,
    tag: str,
    desc: str,
) -> None:

    base_path = f"{tag}_{desc}"

    blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=cast(str, os.getenv("POLYGON_REFERENCE_STORAGE_CONNTECTION_STRING"))
    )

    blob_location = BlobStorageLocation(
        blobContainer="kaa",
        blobName=f"{base_path}/reference_image.fff",
    )
    thermal_image_array = download_fff_image(
        blob_service_client=blob_service_client, blob_storage_location=blob_location
    )

    try:
        blob_location.blob_name = f"{base_path}/reference_image.jpeg"
        jpeg_array = download_blob_to_image(
            blob_service_client=blob_service_client, blob_storage_location=blob_location
        )

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(jpeg_array, cmap="jet")
        plt.subplot(1, 2, 2)
        plt.imshow(thermal_image_array, cmap="jet")
        ax = plt.subplot(1, 2, 2)

    except Exception as e:
        logger.warning(f"Failed to load JPEG image: {e}. Using FFF image only.")

        fig = plt.figure()
        plt.imshow(thermal_image_array, cmap="jet")
        ax = plt.gca()

    # Draw polygon interactively
    polygon_points = np.empty((0, 2), dtype=int)
    patch = plt.Polygon(
        polygon_points, closed=True, fill=None, edgecolor="r", linewidth=1
    )
    ax.add_patch(patch)

    klicker0 = clicker(ax, classes=["image0"], markers=["x"], colors=["red"])
    klicker0.set_positions({"image0": []})
    klicker0._update_points()

    def on_click(event: Any) -> None:
        points = klicker0.get_positions()["image0"]
        patch.set_xy(points)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Save polygon when the figure is closed
    def on_close(event: Any) -> None:
        try:
            points = klicker0.get_positions().get("image0", [])
            print(points)
            if points is None:
                logger.info("No polygon points to save on close.")
                return

            points_arr = np.asarray(points)
            if points_arr.size == 0:
                logger.info("No polygon points to save on close.")
                return

            # Normalize to 2D array of shape (N, 2)
            points_arr = np.atleast_2d(points_arr)
            if points_arr.shape[1] != 2:
                logger.error(
                    f"Unexpected polygon array shape on close: {points_arr.shape}. Expected Nx2."
                )
                return

            polygon_list = points_arr.astype(int).tolist()

            # Upload polygon JSON back to blob storage
            polygon_json = json.dumps(polygon_list, indent=4)
            from io import BytesIO

            buf = BytesIO()
            buf.write(polygon_json.encode("utf-8"))
            buf.seek(0)

            # Use the provided storage_account as the blob container and upload
            polygon_blob_name = f"{base_path}/reference_polygon.json"
            polygon_blob_location = BlobStorageLocation(
                blobContainer="kaa",
                blobName=polygon_blob_name,
            )

            try:
                upload_bytes_to_blob(
                    blob_service_client,
                    polygon_blob_location,
                    buf,
                    content_type="application/json",
                )
                logger.info(
                    f"Uploaded polygon JSON to blob: {polygon_blob_location.blob_container}/{polygon_blob_location.blob_name}"
                )
            except Exception as e:
                logger.error(f"Failed to upload polygon JSON to blob: {e}")
        except Exception as e:
            logger.error(f"Failed to save polygon on close: {e}")

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()
