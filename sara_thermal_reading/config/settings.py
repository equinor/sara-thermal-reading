from enum import Enum

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class ImageAlignmentMethod(str, Enum):
    ORB_BF_CV2 = "ORB_BF_CV2"
    WARP_POLYGON = "WARP_POLYGON"


class Settings(BaseSettings):
    SOURCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    DESTINATION_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_IMAGE_FILENAME: str = Field(default="reference_image.fff")
    REFERENCE_POLYGON_FILENAME: str = Field(default="reference_polygon.json")
    IMAGE_ALIGNMENT_METHOD: ImageAlignmentMethod = Field(
        default=ImageAlignmentMethod.ORB_BF_CV2
    )
    WORKFLOW_TO_RUN: str = Field(default="fff-workflow")
    OTEL_SERVICE_NAME: str = Field(default="sara-thermal-reading")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_EXPORTER_OTLP_PROTOCOL: str = Field(default="grpc")


settings = Settings()
