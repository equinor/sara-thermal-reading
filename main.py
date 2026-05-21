import logging

import typer
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from sara_thermal_reading.cli_inputs import (
    parse_extras,
    parse_input_blob_storage_locations,
    parse_output_blob_storage_location,
)
from sara_thermal_reading.config.logger import setup_logger
from sara_thermal_reading.config.open_telemetry import setup_open_telemetry
from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.main_thermal_workflow import run_thermal_reading_workflow
from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation

setup_logger()
logger = logging.getLogger(__name__)
setup_open_telemetry()
tracer = trace.get_tracer(settings.OTEL_SERVICE_NAME)


class ExtrasModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reference_image_blob_storage_location: BlobStorageLocation = Field(
        ..., alias="referenceImageBlobStorageLocation"
    )
    reference_polygon_blob_storage_location: BlobStorageLocation = Field(
        ..., alias="referencePolygonBlobStorageLocation"
    )


app = typer.Typer()


@app.command()
def run_thermal_reading(
    input_blob_storage_locations: str = typer.Option(
        ...,
        help=(
            "JSON array of blob locations to analyze. sara-thermal-reading "
            "is a single-image analyzer, so the array must contain exactly "
            "one entry."
        ),
    ),
    output_blob_storage_location: str = typer.Option(
        ...,
        help="JSON object describing the destination blob for the visualized image.",
    ),
    extras: str = typer.Option(
        ...,
        help=(
            "JSON object with extra per-workflow parameters. For "
            "sara-thermal-reading, must contain "
            "'referenceImageBlobStorageLocation' and "
            "'referencePolygonBlobStorageLocation'."
        ),
    ),
    result_output_file: str = typer.Option(
        "/tmp/result.json",
        help=(
            "Path to write the workflow result JSON to. The file is created "
            "after the workflow succeeds and contains 'temperature' (float)."
        ),
    ),
) -> None:
    anonymized_location = parse_input_blob_storage_locations(
        input_blob_storage_locations
    )[0]
    visualized_location = parse_output_blob_storage_location(
        output_blob_storage_location
    )
    parsed_extras = parse_extras(extras, ExtrasModel)
    reference_image_location = parsed_extras.reference_image_blob_storage_location
    reference_polygon_location = parsed_extras.reference_polygon_blob_storage_location

    with tracer.start_as_current_span(
        "cli.run",
        attributes={
            "src.container": anonymized_location.blob_container,
            "src.blob": anonymized_location.blob_name,
            "dst.container": visualized_location.blob_container,
            "dst.blob": visualized_location.blob_name,
            "reference.image.container": reference_image_location.blob_container,
            "reference.image.blob": reference_image_location.blob_name,
            "reference.polygon.container": reference_polygon_location.blob_container,
            "reference.polygon.blob": reference_polygon_location.blob_name,
        },
    ) as span:
        try:
            run_thermal_reading_workflow(
                anonymized_location,
                visualized_location,
                reference_image_location,
                reference_polygon_location,
                result_output_file,
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise


if __name__ == "__main__":
    app()
