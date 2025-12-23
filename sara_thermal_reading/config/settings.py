from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    SOURCE_STORAGE_ACCOUNT: str = Field(default="")
    SOURCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    DESTINATION_STORAGE_ACCOUNT: str = Field(default="")
    DESTINATION_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_STORAGE_ACCOUNT: str = Field(default="")
    REFERENCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    REFERENCE_IMAGE_FILENAME: str = Field(default="reference_image.jpeg")


settings = Settings()
