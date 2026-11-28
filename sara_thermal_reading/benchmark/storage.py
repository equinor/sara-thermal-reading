import json
import os
import re
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential, WorkloadIdentityCredential
from azure.storage.blob import ContainerClient, ContentSettings


class BenchmarkStorage:
    def __init__(self, location: str, use_default_credential: bool = False) -> None:
        self.root = Path(location)
        self.blob: ContainerClient | None = None
        if "://" in location:
            if not re.fullmatch(
                r"https://[a-z0-9]{3,24}\.blob\.core\.windows\.net/[a-z0-9-]+/?",
                location,
            ):
                raise ValueError(
                    "Expected an Azure Blob container URL without credentials"
                )
            connection_string = os.environ.get("BENCHMARK_STORAGE_CONNECTION_STRING")
            if connection_string:
                container = location.rstrip("/").rsplit("/", 1)[-1]
                self.blob = ContainerClient.from_connection_string(
                    connection_string, container_name=container
                )
            else:
                credential: Any = (
                    DefaultAzureCredential()
                    if use_default_credential
                    else WorkloadIdentityCredential()
                )
                self.blob = ContainerClient.from_container_url(location, credential)

    @property
    def name(self) -> str:
        if self.blob is not None:
            return str(self.blob.container_name)
        return self.root.resolve().name

    def exists(self, name: str) -> bool:
        if self.blob is not None:
            return bool(self.blob.get_blob_client(name).exists())
        return (self.root / name).exists()

    def read(self, name: str) -> bytes:
        if self.blob is not None:
            return bytes(self.blob.download_blob(name).readall())
        return (self.root / name).read_bytes()

    def list_names(self, prefix: str = "") -> list[str]:
        if self.blob is not None:
            return sorted(
                str(blob.name) for blob in self.blob.list_blobs(name_starts_with=prefix)
            )
        if not self.root.is_dir():
            return []
        results: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if rel.startswith(prefix):
                results.append(rel)
        return sorted(results)

    def create_json(self, name: str, value: object) -> None:
        self.write_json(name, value, overwrite=False)

    def write_json(self, name: str, value: object, overwrite: bool = True) -> None:
        content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self.write_bytes(
            name, content, content_type="application/json", overwrite=overwrite
        )

    def create_bytes(
        self,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.write_bytes(name, content, content_type=content_type, overwrite=False)

    def write_bytes(
        self,
        name: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> None:
        if self.blob is not None:
            self.blob.upload_blob(
                name,
                content,
                overwrite=overwrite,
                content_settings=ContentSettings(content_type=content_type),
            )
            return
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if overwrite else "xb"
        with path.open(mode) as file:
            file.write(content)
