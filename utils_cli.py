from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import typer
from azure.storage.blob import BlobServiceClient

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_cloud_reference_polygon,
    create_reference_polygon,
    show_cloud_reference_polygon,
)
from sara_thermal_reading.dev_utils.run_fff_workflow_local_files import (
    run_fff_workflow_local_files,
)
from sara_thermal_reading.file_io.fff_loader import load_fff_from_bytes
from sara_thermal_reading.file_io.file_utils import load_reference_fff_image_and_polygon
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
    image, polygon_points = load_reference_fff_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    plot_thermal_image(
        image,
        f"Reference Image: {installation_code}/{tag_id}_{inspection_description}",
        polygon_points,
    )
    plt.show()


@app.command()
def create_polygon(
    fff_image_path: Path = typer.Argument(
        ..., help="Path to the thermal image (FFF file)"
    ),
    polygon_json_output_path: Path = typer.Option(
        Path("reference_polygon.json"), help="Path to save the polygon JSON"
    ),
) -> None:
    create_reference_polygon(fff_image_path, polygon_json_output_path)


@app.command()
def create_polygon_cloud(
    storage_account: str = typer.Argument(..., help="Storage account name"),
    tag: str = typer.Argument(..., help="Tag"),
    desc: str = typer.Argument(..., help="Inspection description"),
) -> None:
    create_cloud_reference_polygon(
        storage_account,
        tag,
        desc,
    )


@app.command()
def show_polygon_cloud(
    storage_account: str = typer.Argument(..., help="Storage account name"),
    tag: str = typer.Argument(..., help="Tag"),
    desc: str = typer.Argument(..., help="Inspection description"),
) -> None:
    show_cloud_reference_polygon(
        storage_account,
        tag,
        desc,
    )


@app.command()
def view_all_thermal_from_blobs() -> None:
    blob_service_client = BlobServiceClient.from_connection_string(
        settings.SOURCE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client("kaa")
    all_blobs_iterator = container_client.list_blobs()
    all_fff_blobs = [blob for blob in all_blobs_iterator if blob.name.endswith(".fff")]

    for fff_blob in all_fff_blobs:
        blob_client = container_client.get_blob_client(fff_blob.name)
        fff_bytes = blob_client.download_blob().readall()
        fff_image = load_fff_from_bytes(fff_bytes)
        time_str = blob_client.blob_name.split("__")[-1].replace(".fff", "")
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

        ambient_temp_in_image = np.median(fff_image)
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
        plt.imshow(fff_image, cmap="jet")
        plt.axis("off")
        plt.colorbar(label="Temperature (°C)")
        plt.subplot(2, 1, 2)
        plt.hist(
            fff_image.flatten(),
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
            f"./results/{fff_blob.name.replace('/', '_').replace('.fff', '')}_visualization.png",
            data_rgb,
        )
    pass


if __name__ == "__main__":
    app()
