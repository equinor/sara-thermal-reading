"""Typer callbacks for parsing the generic SARA blob-location CLI contract.

SARA invokes analyzer images with three JSON-encoded options:

* ``--input-blob-storage-locations``  -- a JSON array of blob locations.
* ``--output-blob-storage-location``  -- a single JSON object.
* ``--extras``                        -- a JSON object of per-workflow-type
  custom fields. The schema is owned by each image (see ``ExtrasModel``).

sara-thermal-reading is a single-image analyzer, so the input array must
contain exactly one element. Anything else is a hard failure surfaced
through Typer.
"""

import json
from typing import List, Type, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation

ExtrasT = TypeVar("ExtrasT", bound=BaseModel)


def parse_input_blob_storage_locations(value: str) -> List[BlobStorageLocation]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--input-blob-storage-locations is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise typer.BadParameter(
            "--input-blob-storage-locations must be a JSON array of blob "
            f"locations, got {type(raw).__name__}."
        )

    if len(raw) != 1:
        raise typer.BadParameter(
            "--input-blob-storage-locations must contain exactly one entry "
            f"(sara-thermal-reading is a single-image analyzer); got {len(raw)}."
        )

    try:
        return [BlobStorageLocation.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise typer.BadParameter(
            f"--input-blob-storage-locations entry is invalid: {exc}"
        ) from exc


def parse_output_blob_storage_location(value: str) -> BlobStorageLocation:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--output-blob-storage-location is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise typer.BadParameter(
            "--output-blob-storage-location must be a JSON object, "
            f"got {type(raw).__name__}."
        )

    try:
        return BlobStorageLocation.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"--output-blob-storage-location is invalid: {exc}"
        ) from exc


def parse_extras(value: str, model: Type[ExtrasT]) -> ExtrasT:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--extras is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"--extras must be a JSON object, got {type(raw).__name__}."
        )

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"--extras is invalid: {exc}") from exc
