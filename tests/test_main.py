import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

with patch("sara_thermal_reading.config.open_telemetry.setup_open_telemetry"):
    import main


def test_cli_passes_inline_polygon_to_workflow() -> None:
    location = {
        "storageAccount": "account",
        "blobContainer": "kaa",
        "blobName": "image.tiff",
    }
    polygon = [{"x": 287.5, "y": 137}, {"x": 277, "y": 186}, {"x": 360, "y": 194}]
    with patch.object(main, "run_thermal_reading_workflow") as workflow:
        result = CliRunner().invoke(
            main.app,
            [
                "--input-blob-storage-locations",
                json.dumps([location]),
                "--output-blob-storage-location",
                json.dumps(location),
                "--extras",
                json.dumps(
                    {
                        "referenceImageBlobStorageLocation": location,
                        "referencePolygon": polygon,
                    }
                ),
            ],
        )

    assert result.exit_code == 0, result.output
    workflow.assert_called_once()
    assert workflow.call_args.args[3] == [(287.5, 137), (277, 186), (360, 194)]


@pytest.mark.parametrize(
    "polygon",
    [
        [],
        [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
        [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": float("nan"), "y": 2}],
        [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 2}],
    ],
)
def test_cli_rejects_invalid_inline_polygon(polygon: list[dict[str, float]]) -> None:
    location = {
        "storageAccount": "account",
        "blobContainer": "kaa",
        "blobName": "image.tiff",
    }
    with patch.object(main, "run_thermal_reading_workflow") as workflow:
        result = CliRunner().invoke(
            main.app,
            [
                "--input-blob-storage-locations",
                json.dumps([location]),
                "--output-blob-storage-location",
                json.dumps(location),
                "--extras",
                json.dumps(
                    {
                        "referenceImageBlobStorageLocation": location,
                        "referencePolygon": polygon,
                    }
                ),
            ],
        )

    assert result.exit_code == 2
    assert "referencePolygon" in result.output
    workflow.assert_not_called()
