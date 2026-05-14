from pydantic import BaseModel, ConfigDict, Field, field_validator


class BlobStorageLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    blob_container: str = Field(..., alias="blobContainer")
    blob_name: str = Field(..., alias="blobName")

    @field_validator("blob_container")
    def validate_blob_container(cls, v: str) -> str:
        if not v:
            raise ValueError("blobContainer cannot be empty")
        return v

    @field_validator("blob_name")
    def validate_blob_name(cls, v: str) -> str:
        if not v:
            raise ValueError("blobName cannot be empty")
        return v
