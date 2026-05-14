import json

import pytest
import typer
from pydantic import BaseModel, ConfigDict

from sara_thermal_reading.cli_inputs import (
    parse_extras,
    parse_input_blob_storage_locations,
    parse_output_blob_storage_location,
)
from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation


class _ClosedExtras(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _loc(container: str = "b", blob: str = "c") -> dict[str, str]:
    return {
        "blobContainer": container,
        "blobName": blob,
    }


def test_parse_input_returns_single_blob_storage_location() -> None:
    expected: list[dict[str, str]] = [
        {
            "blobContainer": "src-container",
            "blobName": "src-blob.tiff",
        }
    ]
    payload: str = json.dumps(expected)

    result = parse_input_blob_storage_locations(payload)

    assert len(result) == len(expected)
    location: BlobStorageLocation = result[0]
    assert location.blob_container == expected[0]["blobContainer"]
    assert location.blob_name == expected[0]["blobName"]


def test_parse_output_returns_blob_storage_location() -> None:
    expected: dict[str, str] = {
        "blobContainer": "dst-container",
        "blobName": "dst-blob.png",
    }
    payload: str = json.dumps(expected)

    result: BlobStorageLocation = parse_output_blob_storage_location(payload)

    assert result.blob_container == expected["blobContainer"]
    assert result.blob_name == expected["blobName"]


@pytest.mark.parametrize(
    "payload, expected_message_fragment",
    [
        ("not json{", "valid JSON"),
        (json.dumps({"blobContainer": "a"}), "JSON array"),
        (json.dumps([]), "exactly one"),
        (json.dumps([_loc(), _loc("d", "e")]), "exactly one"),
        (json.dumps([{"blobContainer": "a"}]), None),
    ],
    ids=[
        "malformed-json",
        "object-instead-of-array",
        "empty-array",
        "multiple-entries",
        "missing-required-field",
    ],
)
def test_parse_input_rejects_invalid_payloads(
    payload: str, expected_message_fragment: str | None
) -> None:
    with pytest.raises(typer.BadParameter, match=expected_message_fragment):
        parse_input_blob_storage_locations(payload)


@pytest.mark.parametrize(
    "payload, expected_message_fragment",
    [
        ("not json{", "valid JSON"),
        (json.dumps([_loc()]), "JSON object"),
        (json.dumps({"blobContainer": "a"}), None),
    ],
    ids=[
        "malformed-json",
        "array-instead-of-object",
        "missing-required-field",
    ],
)
def test_parse_output_rejects_invalid_payloads(
    payload: str, expected_message_fragment: str | None
) -> None:
    with pytest.raises(typer.BadParameter, match=expected_message_fragment):
        parse_output_blob_storage_location(payload)


def test_parse_extras_returns_validated_model_instance() -> None:
    result = parse_extras("{}", _ClosedExtras)

    assert isinstance(result, _ClosedExtras)


@pytest.mark.parametrize(
    "payload, expected_message_fragment",
    [
        ("not json{", "valid JSON"),
        (json.dumps([]), "JSON object"),
        (json.dumps({"unexpected": "field"}), "invalid"),
    ],
    ids=[
        "malformed-json",
        "array-instead-of-object",
        "unknown-key-rejected",
    ],
)
def test_parse_extras_rejects_invalid_payloads(
    payload: str, expected_message_fragment: str | None
) -> None:
    with pytest.raises(typer.BadParameter, match=expected_message_fragment):
        parse_extras(payload, _ClosedExtras)
