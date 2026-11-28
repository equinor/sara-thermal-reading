import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import tifffile

from sara_thermal_reading.benchmark.storage import BenchmarkStorage
from sara_thermal_reading.benchmark.thermal_split import (
    CONTAINER_SCHEMA_FILENAME,
    THERMAL_DATASET_FORMAT,
    THERMAL_SCHEMA_VERSION,
    ThermalSplit,
)
from sara_thermal_reading.dev_utils.dataset_builder import (
    SOURCE_FILES_PREFIX,
    BlobThermalDataset,
    SourceFileGroup,
)

EXAMPLE_DATA = Path(__file__).parent.parent / "example-data" / "asset-example"


def _tiff_bytes(image: np.ndarray) -> bytes:
    buf = BytesIO()
    tifffile.imwrite(buf, image.astype(np.float64))
    return buf.getvalue()


def _build_example_dataset(dataset_root: Path, version: str = "v001") -> Path:
    ref_tiff_path = EXAMPLE_DATA / "thermal_image.tiff"
    ref_bytes = ref_tiff_path.read_bytes()
    ref_image = tifffile.imread(ref_tiff_path).astype(np.float64)
    ref_polygon = [
        (float(x), float(y))
        for x, y in json.loads((EXAMPLE_DATA / "polygon.json").read_text())
    ]

    storage = BenchmarkStorage(str(dataset_root))
    storage.write_json(
        CONTAINER_SCHEMA_FILENAME,
        {"format": THERMAL_DATASET_FORMAT, "schema_version": THERMAL_SCHEMA_VERSION},
        overwrite=True,
    )
    ref_blob = f"{SOURCE_FILES_PREFIX}/TAG-1__ThermalImage__bearing__20260101.tiff"
    sample_blob = f"{SOURCE_FILES_PREFIX}/TAG-1__ThermalImage__bearing__20260102.tiff"
    storage.write_bytes(ref_blob, ref_bytes, content_type="image/tiff", overwrite=True)
    storage.write_bytes(
        sample_blob, _tiff_bytes(ref_image), content_type="image/tiff", overwrite=True
    )

    dataset = BlobThermalDataset(storage, version=version, split="val")
    group = SourceFileGroup(
        group_id="TAG-1__bearing",
        tag_id="TAG-1",
        inspection_description="bearing",
    )
    ref = dataset.add_reference(group, ref_blob, ref_image, ref_polygon)
    dataset.add_sample(
        reference_id=ref.reference_id,
        source_tiff_blob=sample_blob,
        image=ref_image,
        matchable=True,
        gt_polygon=ref_polygon,
    )
    return dataset_root


def test_thermal_split_fails_when_schema_json_missing(tmp_path: Path) -> None:
    _build_example_dataset(tmp_path)
    (tmp_path / CONTAINER_SCHEMA_FILENAME).unlink()

    storage = BenchmarkStorage(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ThermalSplit.load(storage.read, version="v001", split="val")


def test_thermal_split_fails_when_schema_format_is_not_thermal(tmp_path: Path) -> None:
    _build_example_dataset(tmp_path)
    (tmp_path / CONTAINER_SCHEMA_FILENAME).write_text(
        json.dumps({"format": "coco", "schema_version": 1}),
        encoding="utf-8",
    )

    storage = BenchmarkStorage(str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        ThermalSplit.load(storage.read, version="v001", split="val")
