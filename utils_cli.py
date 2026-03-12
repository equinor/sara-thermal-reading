from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import typer

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_reference_polygon,
)
from sara_thermal_reading.dev_utils.run_fff_workflow_local_files import (
    run_fff_workflow_local_files,
)
from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.visualization.plotting import (
    plot_fff_from_path,
    plot_thermal_image,
)

app = typer.Typer()


@app.command()
def run_fff_workflow(
    polygon_path: str = typer.Option(..., help="Path to the polygon JSON file"),
    reference_image_path: str = typer.Option(
        ..., help="Path to the reference FFF image"
    ),
    source_image_path: str = typer.Option(
        None, help="Path to the source FFF image (optional)"
    ),
    tag_id: str = typer.Option("test_tag_id", help="Tag ID"),
    inspection_description: str = typer.Option(
        "test_inspection_description", help="Inspection description"
    ),
) -> None:
    run_fff_workflow_local_files(
        polygon_path,
        reference_image_path,
        source_image_path,
        tag_id,
        inspection_description,
    )


@app.command()
def plot_fff(
    file_path: Path,
    polygon_json_path: Path = typer.Option(
        None, help="Path to the polygon JSON file to plot"
    ),
) -> None:
    plot_fff_from_path(file_path, polygon_json_path)


@app.command()
def plot_current_reference_image_and_polygon(
    installation_code: str = typer.Option(..., help="Installation code"),
    tag_id: str = typer.Option(..., help="Tag ID"),
    inspection_description: str = typer.Option(..., help="Inspection description"),
) -> None:
    reference_blob_store = BlobStore(
        installation_code=installation_code,
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    reference_image_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_FFF_FILENAME}"
    )
    reference_image_fff: np.ndarray = reference_blob_store.download_fff(
        reference_image_blob_name
    )
    reference_polygon_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_POLYGON_FILENAME}"
    )
    reference_polygon: list[tuple[int, int]] = reference_blob_store.download_polygon(
        reference_polygon_blob_name
    )

    plot_thermal_image(
        reference_image_fff,
        f"Reference Image: {installation_code}/{tag_id}_{inspection_description}",
        reference_polygon,
    )
    plt.show()


@app.command()
def create_polygon_cloud(
    tag_id: str = typer.Option(..., help="Tag"),
    inspection_description: str = typer.Option(..., help="Inspection description"),
) -> None:
    reference_blob_store = BlobStore(
        installation_code="kaa",
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    reference_image_fff_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_FFF_FILENAME}"
    )
    reference_image_fff: np.ndarray = reference_blob_store.download_fff(
        reference_image_fff_blob_name
    )
    reference_polygon_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_POLYGON_FILENAME}"
    )
    reference_polygon: list[tuple[int, int]] | None
    if reference_blob_store.check_if_exists(reference_polygon_blob_name):
        reference_polygon = reference_blob_store.download_polygon(
            reference_polygon_blob_name
        )
    else:
        reference_polygon = None
    reference_image_jpg_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_JPG_FILENAME}"
    )
    reference_image_jpg: np.ndarray | None
    if reference_blob_store.check_if_exists(reference_image_jpg_blob_name):
        reference_image_jpg = reference_blob_store.download_image_array(
            reference_image_jpg_blob_name
        )
    else:
        reference_image_jpg = None

    polygon: list[tuple[int, int]] | None = create_reference_polygon(
        ref_image_fff=reference_image_fff,
        ref_image_jpg=reference_image_jpg,
        ref_polygon=reference_polygon,
    )

    if polygon is not None:
        reference_blob_store.upload_polygon(polygon, reference_polygon_blob_name)


def plt_polygon(
    polygon: list[tuple[int, int]],
) -> None:
    x_coordinates = [point[0] for point in polygon]
    y_coordinates = [point[1] for point in polygon]
    plt.fill(x_coordinates, y_coordinates, edgecolor="r", fill=False, linewidth=1)


@app.command()
def view_all_thermal_from_blobs() -> None:
    source_blob_store = BlobStore(
        installation_code="kaa",
        connection_string=settings.SOURCE_STORAGE_CONNECTION_STRING,
    )
    all_blob_names: list[str] = source_blob_store.list_blobs_by_prefix(prefix="")
    all_fff_blob_names: list[str] = [
        name for name in all_blob_names if name.endswith(".fff")
    ]

    for fff_blob_name in all_fff_blob_names:
        image_fff = source_blob_store.download_fff(fff_blob_name)
        time_str = fff_blob_name.split("__")[-1].replace(".fff", "")
        time_datetime = datetime.strptime(time_str, "%Y%m%d-%H%M%S")

        year, month, day, hour = (
            time_datetime.year,
            time_datetime.month,
            time_datetime.day,
            time_datetime.hour + 1,
        )
        url = f"https://rim.k8s.met.no/api/v1/observations?sources=SN47260&referenceTime={year}-{month:02d}-{day:02d}T{hour:02d}:00:00Z/{year}-{month:02d}-{day:02d}T{hour:02d}:59:59Z&elements=air_temperature&timeResolution=hours"
        response = requests.get(url, allow_redirects=False)
        ambient_temp_from_api = response.json()["data"][0]["observations"][0]["value"]
        print(f"Ambient temperature from API: {ambient_temp_from_api:.2f} °C")

        ambient_temp_in_image = np.median(image_fff)
        print(f"Ambient temperature measured in image: {ambient_temp_in_image:.2f} °C")

        plt.figure()
        plt.clf()
        fig = plt.gcf()
        fig.set_figwidth(8)
        fig.set_figheight(9)
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=1, wspace=None, hspace=None
        )
        plt.subplot(2, 1, 1)
        plt.imshow(image_fff, cmap="jet")
        plt.axis("off")
        plt.colorbar(label="Temperature (°C)")
        plt.subplot(2, 1, 2)
        plt.hist(
            image_fff.flatten(),
            bins=100,
            range=(ambient_temp_in_image - 5, ambient_temp_in_image + 5),
        )
        plt.axvline(
            ambient_temp_in_image,
            color="black",
            linestyle="dashed",
            linewidth=2,
            label=f"Image: {ambient_temp_in_image:.2f} °C",
        )
        plt.axvline(
            ambient_temp_from_api,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"API: {ambient_temp_from_api:.2f} °C",
        )
        plt.legend()
        plt.show()

        fig.canvas.draw()
        # Mypy does not see that canvas is FigureCanvasAgg and buffer_rgba exists
        data_rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
        data_rgb = data_rgba[:, :, :3]

        plt.imsave(
            f"./results/{fff_blob_name.replace('/', '_').replace('.fff', '')}_visualization.png",
            data_rgb,
        )
    pass


if __name__ == "__main__":
    app()
