from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    SOURCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    DESTINATION_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_IMAGE_TIFF_FILENAME: str = Field(default="reference_image.tiff")
    REFERENCE_IMAGE_JPG_FILENAME: str = Field(default="reference_image.jpeg")
    REFERENCE_POLYGON_FILENAME: str = Field(default="reference_polygon.json")
    TEMPERATURE_PERCENTILE: float = Field(default=95.0)
    MIN_ALIGNMENT_SCORE: float = Field(default=0.1)
    OTEL_SERVICE_NAME: str = Field(default="sara-thermal-reading")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_EXPORTER_OTLP_PROTOCOL: str = Field(default="grpc")


settings = Settings()
