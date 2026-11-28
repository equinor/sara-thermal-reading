import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import tifffile
import typer

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.dev_utils.create_reference_polygon import (
    create_reference_polygon,
)
from sara_thermal_reading.dev_utils.dataset_builder import (
    DEFAULT_CONTAINER,
    DEFAULT_STORAGE_ACCOUNT,
    get_dataset_status,
    label_dataset,
    print_dataset_status,
    upload_source_files,
    validate_blob_dataset,
)
from sara_thermal_reading.dev_utils.run_thermal_workflow_local_files import (
    run_thermal_workflow_local_files,
)
from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.main_thermal_workflow import process_thermal_image
from sara_thermal_reading.visualization.plotting import plot_thermal_from_path

app = typer.Typer()

REFERENCE_POLYGON_FILENAME = "reference_polygon.json"


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
    run_thermal_workflow_local_files(
        polygon_path,
        reference_image_path,
        source_image_path,
        tag_id,
        inspection_description,
    )


@app.command()
def plot_thermal(
    file_path: Path,
    polygon_json_path: Path = typer.Option(
        None, help="Path to the polygon JSON file to plot"
    ),
) -> None:
    plot_thermal_from_path(file_path, polygon_json_path)


@app.command()
def create_polygon_local(
    image_path: Path = typer.Option(..., help="Path to local thermal TIFF image"),
    output_path: Path = typer.Option(..., help="Path to save the polygon JSON file"),
) -> None:
    """Create a reference polygon by clicking on a local thermal TIFF image."""
    image: np.ndarray = tifffile.imread(image_path)

    existing_polygon: list[tuple[int, int]] | None = None
    if output_path.exists():
        with open(output_path) as f:
            existing_polygon = [tuple(p) for p in json.load(f)]

    polygon = create_reference_polygon(
        ref_image=image,
        ref_image_jpg=None,
        ref_polygon=existing_polygon,
    )

    if polygon is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(polygon, f)
        print(f"Polygon saved to {output_path}")
    else:
        print("No polygon created (less than 3 points).")


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
    all_tiff_blob_names: list[str] = [
        name for name in all_blob_names if name.endswith(".tiff")
    ]

    for tiff_blob_name in all_tiff_blob_names:
        image = source_blob_store.download_thermal_tiff(tiff_blob_name)
        time_str = tiff_blob_name.split("__")[-1].replace(".tiff", "")
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

        ambient_temp_in_image = np.median(image)
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
        plt.imshow(image, cmap="jet")
        plt.axis("off")
        plt.colorbar(label="Temperature (°C)")
        plt.subplot(2, 1, 2)
        plt.hist(
            image.flatten(),
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
            f"./results/{tiff_blob_name.replace('/', '_').replace('.tiff', '')}_visualization.png",
            data_rgb,
        )
    pass


@app.command()
def run_matching_single(
    blob_name: str = typer.Option(..., help="Name of the blob"),
    installation_code: str = typer.Option(
        ..., help="Installation code = name of the container"
    ),
) -> None:
    tag_id: str = blob_name.split("/")[-1].split("__")[0]
    inspection_description: str = blob_name.split("/")[-1].split("__")[2]
    inspection_description = inspection_description.replace("-", " ")

    source_blob_store = BlobStore(
        installation_code=installation_code,
        connection_string=settings.SOURCE_STORAGE_CONNECTION_STRING,
    )
    source_image: np.ndarray = source_blob_store.download_thermal_tiff(blob_name)

    reference_blob_store = BlobStore(
        installation_code=installation_code,
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    reference_image_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_TIFF_FILENAME}"
    )
    reference_image: np.ndarray = reference_blob_store.download_thermal_tiff(
        reference_image_blob_name
    )
    reference_polygon_blob_name = (
        f"{tag_id}_{inspection_description}/{REFERENCE_POLYGON_FILENAME}"
    )
    reference_polygon: list[tuple[int, int]] = reference_blob_store.download_polygon(
        reference_polygon_blob_name
    )

    (
        temperature,
        annotated_image,
        warped_polygon_list,
        warped_reference_img,
        _,
    ) = process_thermal_image(reference_image, source_image, reference_polygon)

    plot_matching(
        reference_image=reference_image,
        source_image=source_image,
        reference_polygon=reference_polygon,
        warped_polygon_list=warped_polygon_list,
    )


def plot_matching(
    reference_image: np.ndarray,
    source_image: np.ndarray,
    reference_polygon: list[tuple[int, int]],
    warped_polygon_list: list[tuple[int, int]],
) -> None:
    polygon_np = np.array(reference_polygon)
    warped_polygon_np = np.array(warped_polygon_list)

    plt.ioff()

    plt.figure("Matching")
    fig = plt.gcf()
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.title("Reference image")
    plt.imshow(reference_image, cmap="jet")
    plt.fill(
        polygon_np[:, 0],
        polygon_np[:, 1],
        facecolor="red",
        edgecolor="white",
        linewidth=2,
        alpha=0.5,
    )
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Source image")
    plt.imshow(source_image, cmap="jet")
    plt.fill(
        warped_polygon_np[:, 0],
        warped_polygon_np[:, 1],
        facecolor="red",
        edgecolor="white",
        linewidth=2,
        alpha=0.5,
    )
    plt.axis("off")
    plt.colorbar()

    plt.show()


@app.command()
def run_matching_all(
    tag_id: str = typer.Option(..., help="Tag ID"),
    installation_code: str = typer.Option(
        ..., help="Installation code = name of the container"
    ),
    inspection_description: str = typer.Option(..., help="Inspection description"),
) -> None:
    source_blob_store = BlobStore(
        installation_code=installation_code,
        connection_string=settings.SOURCE_STORAGE_CONNECTION_STRING,
    )
    all_blob_names = source_blob_store.list_blobs_by_prefix(prefix="")
    source_tiff_blob_names = [
        name
        for name in all_blob_names
        if name.endswith(".tiff")
        and name.split("/")[-1].split("__")[0] == tag_id
        and name.split("/")[-1].split("__")[2]
        == inspection_description.replace(" ", "-")
    ]

    reference_blob_store = BlobStore(
        installation_code=installation_code,
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    reference_image_blob_name = (
        f"{tag_id}_{inspection_description}/{settings.REFERENCE_IMAGE_TIFF_FILENAME}"
    )
    reference_image: np.ndarray = reference_blob_store.download_thermal_tiff(
        reference_image_blob_name
    )
    reference_polygon_blob_name = (
        f"{tag_id}_{inspection_description}/{REFERENCE_POLYGON_FILENAME}"
    )
    reference_polygon: list[tuple[int, int]] = reference_blob_store.download_polygon(
        reference_polygon_blob_name
    )

    for blob_name in source_tiff_blob_names:
        print(f"Processing: {blob_name}")
        source_image: np.ndarray = source_blob_store.download_thermal_tiff(blob_name)

        temperature, annotated_image, warped_polygon_list, _, _ = process_thermal_image(
            reference_image, source_image, reference_polygon
        )

        plot_matching(
            reference_image=reference_image,
            source_image=source_image,
            reference_polygon=reference_polygon,
            warped_polygon_list=warped_polygon_list,
        )


@app.command("upload-source-files")
def upload_source_files_cmd(
    from_st_acc: str | None = typer.Option(
        None, help="Source Azure storage account to copy thermal TIFFs from"
    ),
    from_container: str | None = typer.Option(
        None, help="Source container name (installation code)"
    ),
    prefix: str | None = typer.Option(
        None, help="Source blob prefix (alternative to --folders-file)"
    ),
    folders_file: Path | None = typer.Option(
        None, help="Text file with source blob prefixes (one per line)"
    ),
    from_dir: Path | None = typer.Option(
        None, help="Local directory of thermal TIFF files to upload into source-files/"
    ),
    target_subfolder: str | None = typer.Option(
        None, help="Optional subfolder under source-files/ for grouping"
    ),
    storage_account: str = typer.Option(
        DEFAULT_STORAGE_ACCOUNT, help="Target dataset Azure storage account"
    ),
    container: str = typer.Option(
        DEFAULT_CONTAINER, help="Target dataset blob container"
    ),
) -> None:
    """Step 1: Upload thermal TIFF files into top-level source-files/ in saradatasets."""
    uploaded, skipped = upload_source_files(
        storage_account=storage_account,
        container=container,
        from_storage_account=from_st_acc,
        from_container=from_container,
        prefix=prefix,
        folders_file=folders_file,
        from_dir=from_dir,
        target_subfolder=target_subfolder,
    )
    print(
        f"Uploaded {uploaded} thermal TIFF files to "
        f"https://{storage_account}.blob.core.windows.net/{container}/source-files/ "
        f"(skipped {skipped} existing)."
    )


@app.command()
def dataset_status(
    version: str = typer.Option("v001", help="Dataset version (e.g. v001)"),
    split: str = typer.Option("val", help="Dataset split name (e.g. val)"),
    show_all: bool = typer.Option(
        False, help="List every source file with its status instead of only unlabelled"
    ),
    storage_account: str = typer.Option(
        DEFAULT_STORAGE_ACCOUNT, help="Dataset Azure storage account"
    ),
    container: str = typer.Option(DEFAULT_CONTAINER, help="Dataset blob container"),
) -> None:
    """Show which files in source-files/ are references, labelled, or unlabelled."""
    summary = get_dataset_status(
        version=version,
        split=split,
        storage_account=storage_account,
        container=container,
    )
    print_dataset_status(summary, show_all=show_all)


@app.command("label-dataset")
def label_dataset_cmd(
    version: str = typer.Option("v001", help="Dataset version (e.g. v001)"),
    split: str = typer.Option("val", help="Dataset split name (e.g. val)"),
    from_version: str | None = typer.Option(
        None,
        help="Optional existing version to seed annotations from when creating a new version",
    ),
    group: str | None = typer.Option(
        None, help="Optional substring filter to label only matching tag/group IDs"
    ),
    storage_account: str = typer.Option(
        DEFAULT_STORAGE_ACCOUNT, help="Dataset Azure storage account"
    ),
    container: str = typer.Option(DEFAULT_CONTAINER, help="Dataset blob container"),
) -> None:
    """Step 2: Interactively label unlabelled files from source-files/ directly in blob storage."""
    label_dataset(
        version=version,
        split=split,
        storage_account=storage_account,
        container=container,
        from_version=from_version,
        group_filter=group,
    )


@app.command()
def validate_dataset(
    version: str = typer.Option("v001", help="Dataset version (e.g. v001)"),
    split: str = typer.Option("val", help="Dataset split name (e.g. val)"),
    storage_account: str = typer.Option(
        DEFAULT_STORAGE_ACCOUNT, help="Dataset Azure storage account"
    ),
    container: str = typer.Option(DEFAULT_CONTAINER, help="Dataset blob container"),
) -> None:
    """Validate a thermal validation dataset directly in Azure Blob Storage."""
    thermal_split = validate_blob_dataset(
        version=version,
        split=split,
        storage_account=storage_account,
        container=container,
    )
    matchable = sum(1 for sample in thermal_split.samples.values() if sample.matchable)
    unmatchable = len(thermal_split.samples) - matchable
    print(
        f"Valid thermal dataset ({version}/{split}): "
        f"{len(thermal_split.references)} references, "
        f"{len(thermal_split.samples)} samples "
        f"({matchable} matchable, {unmatchable} unmatchable)."
    )


if __name__ == "__main__":
    app()
