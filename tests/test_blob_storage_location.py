import pytest
from pydantic import ValidationError

from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation


def test_valid_construction_by_alias() -> None:
    """Construction via JSON alias names (blobContainer, blobName) should work."""
    loc = BlobStorageLocation(
        **{"blobContainer": "my-container", "blobName": "my-blob"}
    )

    assert loc.blob_container == "my-container"
    assert loc.blob_name == "my-blob"


def test_valid_construction_by_field_name() -> None:
    """Construction via Python field names should work (populate_by_name=True)."""
    loc = BlobStorageLocation.model_validate(
        {"blob_container": "my-container", "blob_name": "my-blob"}
    )

    assert loc.blob_container == "my-container"
    assert loc.blob_name == "my-blob"


def test_empty_container_rejected() -> None:
    """Empty blobContainer should raise a ValidationError."""
    with pytest.raises(ValidationError, match="blobContainer cannot be empty"):
        BlobStorageLocation(**{"blobContainer": "", "blobName": "my-blob"})


def test_empty_blob_name_rejected() -> None:
    """Empty blobName should raise a ValidationError."""
    with pytest.raises(ValidationError, match="blobName cannot be empty"):
        BlobStorageLocation(**{"blobContainer": "my-container", "blobName": ""})
