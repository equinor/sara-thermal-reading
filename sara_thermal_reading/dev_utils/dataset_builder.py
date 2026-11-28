import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sara_thermal_reading.benchmark.storage import BenchmarkStorage
from sara_thermal_reading.benchmark.thermal_split import (
    CONTAINER_SCHEMA_FILENAME,
    THERMAL_DATASET_FORMAT,
    THERMAL_SCHEMA_VERSION,
    Polygon,
    ThermalSplit,
    validate_container_schema,
)
from sara_thermal_reading.dev_utils.interactive_polygon_labeler import (
    draw_reference_polygon,
    label_thermal_sample,
)
from sara_thermal_reading.dev_utils.thermal_preview import render_thermal_preview_png
from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.file_io.tiff_loader import load_thermal_tiff_from_bytes
from sara_thermal_reading.image_alignment.align_dispatcher import align_thermal_images
from sara_thermal_reading.main_thermal_workflow import (
    preprocess_thermal_pair_for_alignment,
)

DEFAULT_DATASET_ID = "thermal-reading"
DEFAULT_STORAGE_ACCOUNT = "saradatasets"
DEFAULT_CONTAINER = "thermal-reading"
SOURCE_FILES_PREFIX = "source-files"
PREVIEWS_PREFIX = "previews"


@dataclass(frozen=True)
class SourceFileGroup:
    group_id: str
    tag_id: str
    inspection_description: str


@dataclass(frozen=True)
class LoadedReference:
    reference_id: str
    tiff_file: str
    preview_file: str
    image: NDArray[np.float64]
    polygon: Polygon


@dataclass(frozen=True)
class DatasetStatusSummary:
    version: str
    split: str
    total_source_files: int
    reference_files: list[str]
    labelled_matchable: list[str]
    labelled_unmatchable: list[str]
    unlabelled_files: list[str]
    by_group: dict[str, dict[str, list[str]]]


class BlobThermalDataset:
    def __init__(
        self,
        storage: BenchmarkStorage,
        version: str,
        split: str,
        dataset_id: str = DEFAULT_DATASET_ID,
        from_version: str | None = None,
    ) -> None:
        self.storage = storage
        self.version = version
        self.split = split
        self.dataset_id = dataset_id
        self.dataset_prefix = f"{version}/dataset"
        self.annotations_blob = (
            f"{self.dataset_prefix}/annotations/samples_{split}.json"
        )
        self.manifest_blob = f"{self.dataset_prefix}/manifest.json"
        self.references: dict[str, dict[str, Any]] = {}
        self.samples: list[dict[str, Any]] = []
        self._load(from_version=from_version)

    def _load(self, from_version: str | None) -> None:
        if not self.storage.exists(CONTAINER_SCHEMA_FILENAME):
            raise FileNotFoundError(
                f"Missing required '{CONTAINER_SCHEMA_FILENAME}' in container '{self.storage.name}'."
            )
        validate_container_schema(self.storage.read(CONTAINER_SCHEMA_FILENAME))

        if self.storage.exists(self.annotations_blob):
            self._load_annotations_blob(self.annotations_blob)
        elif from_version is not None:
            seed_blob = f"{from_version}/dataset/annotations/samples_{self.split}.json"
            if not self.storage.exists(seed_blob):
                raise FileNotFoundError(f"Seed annotations not found at '{seed_blob}'.")
            self._load_annotations_blob(seed_blob)

    def _load_annotations_blob(self, blob_name: str) -> None:
        data = json.loads(self.storage.read(blob_name))
        for ref in data.get("references", []):
            self.references[str(ref["id"])] = ref
        self.samples = list(data.get("samples", []))

    @property
    def reference_tiff_files(self) -> set[str]:
        return {str(ref["tiff_file"]) for ref in self.references.values()}

    @property
    def labelled_sample_by_tiff(self) -> dict[str, dict[str, Any]]:
        return {str(sample["tiff_file"]): sample for sample in self.samples}

    @property
    def next_sample_id(self) -> int:
        if not self.samples:
            return 1
        return max(int(sample["id"]) for sample in self.samples) + 1

    def get_loaded_reference(self, reference_id: str) -> LoadedReference | None:
        entry = self.references.get(reference_id)
        if entry is None:
            return None
        tiff_file = str(entry["tiff_file"])
        if not self.storage.exists(tiff_file):
            return None
        image = load_thermal_tiff_from_bytes(self.storage.read(tiff_file))
        polygon = [(float(pt[0]), float(pt[1])) for pt in entry["polygon"]]
        return LoadedReference(
            reference_id=reference_id,
            tiff_file=tiff_file,
            preview_file=str(entry["preview_file"]),
            image=image,
            polygon=polygon,
        )

    def ensure_preview(self, source_tiff_blob: str, image: NDArray[np.float64]) -> str:
        preview_blob = preview_path_for_source_blob(source_tiff_blob)
        if not self.storage.exists(preview_blob):
            self.storage.write_bytes(
                preview_blob,
                render_thermal_preview_png(image),
                content_type="image/png",
                overwrite=True,
            )
        return preview_blob

    def add_reference(
        self,
        group: SourceFileGroup,
        source_tiff_blob: str,
        image: NDArray[np.float64],
        polygon: Polygon,
    ) -> LoadedReference:
        preview_blob = self.ensure_preview(source_tiff_blob, image)
        height, width = image.shape[:2]
        self.references[group.group_id] = {
            "id": group.group_id,
            "installation_code": "",
            "tag_id": group.tag_id,
            "inspection_description": group.inspection_description,
            "tiff_file": source_tiff_blob,
            "preview_file": preview_blob,
            "width": width,
            "height": height,
            "polygon": [[float(x), float(y)] for x, y in polygon],
        }
        self.save()
        return LoadedReference(
            reference_id=group.group_id,
            tiff_file=source_tiff_blob,
            preview_file=preview_blob,
            image=image,
            polygon=list(polygon),
        )

    def add_sample(
        self,
        reference_id: str,
        source_tiff_blob: str,
        image: NDArray[np.float64],
        matchable: bool,
        gt_polygon: Sequence[tuple[float, float]] | None,
    ) -> int:
        preview_blob = self.ensure_preview(source_tiff_blob, image)
        sample_id = self.next_sample_id
        height, width = image.shape[:2]
        self.samples.append(
            {
                "id": sample_id,
                "file_name": PurePosixPath(source_tiff_blob).name,
                "tiff_file": source_tiff_blob,
                "preview_file": preview_blob,
                "width": width,
                "height": height,
                "reference_id": reference_id,
                "matchable": matchable,
                "gt_polygon": (
                    [[float(x), float(y)] for x, y in gt_polygon]
                    if matchable and gt_polygon is not None
                    else None
                ),
            }
        )
        self.save()
        return sample_id

    def save(self) -> None:
        self.storage.write_json(
            self.annotations_blob,
            {
                "schema_version": THERMAL_SCHEMA_VERSION,
                "split": self.split,
                "references": list(self.references.values()),
                "samples": self.samples,
            },
            overwrite=True,
        )
        ann_prefix = f"{self.dataset_prefix}/annotations/samples_"
        splits = sorted(
            {
                PurePosixPath(name).stem.removeprefix("samples_")
                for name in self.storage.list_names(ann_prefix)
                if name.endswith(".json")
            }
            | {self.split}
        )
        self.storage.write_json(
            self.manifest_blob,
            {
                "dataset_id": self.dataset_id,
                "version": self.version,
                "format": THERMAL_DATASET_FORMAT,
                "schema_version": THERMAL_SCHEMA_VERSION,
                "splits": splits,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=True,
        )


def open_dataset_storage(
    storage_account: str = DEFAULT_STORAGE_ACCOUNT,
    container: str = DEFAULT_CONTAINER,
) -> BenchmarkStorage:
    container_url = f"https://{storage_account}.blob.core.windows.net/{container}"
    return BenchmarkStorage(container_url, use_default_credential=True)


def upload_source_files(
    storage_account: str = DEFAULT_STORAGE_ACCOUNT,
    container: str = DEFAULT_CONTAINER,
    from_storage_account: str | None = None,
    from_container: str | None = None,
    prefix: str | None = None,
    folders_file: Path | None = None,
    from_dir: Path | None = None,
    target_subfolder: str | None = None,
) -> tuple[int, int]:
    """
    Upload thermal TIFF files into top-level `source-files/` without modifying existing blobs.
    Returns (uploaded_count, skipped_existing_count).
    """
    storage = open_dataset_storage(storage_account, container)
    if not storage.exists(CONTAINER_SCHEMA_FILENAME):
        raise FileNotFoundError(
            f"Missing required '{CONTAINER_SCHEMA_FILENAME}' in container '{container}'."
        )
    validate_container_schema(storage.read(CONTAINER_SCHEMA_FILENAME))

    has_blob_source = from_storage_account is not None or from_container is not None
    has_dir_source = from_dir is not None
    if has_blob_source == has_dir_source:
        raise ValueError(
            "Provide either (--from-st-acc + --from-container) or --from-dir."
        )

    sub_prefix = (
        f"{SOURCE_FILES_PREFIX}/{target_subfolder.strip('/')}"
        if target_subfolder and target_subfolder.strip("/")
        else SOURCE_FILES_PREFIX
    )

    uploaded = 0
    skipped = 0

    if from_dir is not None:
        local_files = sorted(
            path
            for path in from_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in (".tiff", ".tif")
        )
        if not local_files:
            raise FileNotFoundError(f"No TIFF files found in {from_dir}")
        for local_file in local_files:
            rel_posix = local_file.relative_to(from_dir).as_posix()
            target_blob = f"{sub_prefix}/{rel_posix}"
            if storage.exists(target_blob):
                skipped += 1
                continue
            storage.create_bytes(
                target_blob, local_file.read_bytes(), content_type="image/tiff"
            )
            uploaded += 1
        return uploaded, skipped

    if from_storage_account is None or from_container is None:
        raise ValueError("Both --from-st-acc and --from-container are required.")

    prefixes = _resolve_blob_prefixes(folders_file, prefix)
    source_store = BlobStore.from_account_name(from_container, from_storage_account)
    source_blobs = _list_source_thermal_tiffs(source_store, prefixes)

    for source_blob in source_blobs:
        filename = PurePosixPath(source_blob).name
        target_blob = f"{sub_prefix}/{filename}"
        if storage.exists(target_blob):
            skipped += 1
            continue
        data = source_store.download_bytes(source_blob)
        storage.create_bytes(target_blob, data, content_type="image/tiff")
        uploaded += 1

    return uploaded, skipped


def get_dataset_status(
    version: str,
    split: str,
    storage_account: str = DEFAULT_STORAGE_ACCOUNT,
    container: str = DEFAULT_CONTAINER,
) -> DatasetStatusSummary:
    storage = open_dataset_storage(storage_account, container)
    dataset = BlobThermalDataset(storage, version=version, split=split)
    source_blobs = list_container_source_tiffs(storage)
    return summarize_dataset_status(dataset, source_blobs)


def summarize_dataset_status(
    dataset: BlobThermalDataset,
    source_blobs: list[str],
) -> DatasetStatusSummary:
    ref_files = dataset.reference_tiff_files
    samples_by_tiff = dataset.labelled_sample_by_tiff

    references: list[str] = []
    labelled_matchable: list[str] = []
    labelled_unmatchable: list[str] = []
    unlabelled: list[str] = []
    by_group: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {
            "reference": [],
            "labelled_matchable": [],
            "labelled_unmatchable": [],
            "unlabelled": [],
        }
    )

    for blob in source_blobs:
        group = infer_group_for_source_blob(blob)
        bucket = by_group[group.group_id]
        if blob in ref_files:
            references.append(blob)
            bucket["reference"].append(blob)
        elif blob in samples_by_tiff:
            if bool(samples_by_tiff[blob].get("matchable", True)):
                labelled_matchable.append(blob)
                bucket["labelled_matchable"].append(blob)
            else:
                labelled_unmatchable.append(blob)
                bucket["labelled_unmatchable"].append(blob)
        else:
            unlabelled.append(blob)
            bucket["unlabelled"].append(blob)

    return DatasetStatusSummary(
        version=dataset.version,
        split=dataset.split,
        total_source_files=len(source_blobs),
        reference_files=references,
        labelled_matchable=labelled_matchable,
        labelled_unmatchable=labelled_unmatchable,
        unlabelled_files=unlabelled,
        by_group=dict(sorted(by_group.items())),
    )


def print_dataset_status(
    summary: DatasetStatusSummary,
    show_all: bool = False,
) -> None:
    labelled_total = len(summary.labelled_matchable) + len(summary.labelled_unmatchable)
    print(
        f"Dataset {summary.version}/{summary.split}: "
        f"{summary.total_source_files} source files | "
        f"{len(summary.reference_files)} references | "
        f"{labelled_total} labelled "
        f"({len(summary.labelled_matchable)} matchable, "
        f"{len(summary.labelled_unmatchable)} unmatchable) | "
        f"{len(summary.unlabelled_files)} unlabelled"
    )
    for group_id, buckets in summary.by_group.items():
        print(
            f"  [{group_id}] ref={len(buckets['reference'])} "
            f"labelled={len(buckets['labelled_matchable']) + len(buckets['labelled_unmatchable'])} "
            f"unlabelled={len(buckets['unlabelled'])}"
        )
        if show_all:
            for status_key in (
                "reference",
                "labelled_matchable",
                "labelled_unmatchable",
                "unlabelled",
            ):
                for blob in buckets[status_key]:
                    print(f"    {status_key:20s} {blob}")
        else:
            for blob in buckets["unlabelled"]:
                print(f"    unlabelled           {blob}")


def label_dataset(
    version: str,
    split: str,
    storage_account: str = DEFAULT_STORAGE_ACCOUNT,
    container: str = DEFAULT_CONTAINER,
    from_version: str | None = None,
    group_filter: str | None = None,
) -> None:
    storage = open_dataset_storage(storage_account, container)
    dataset = BlobThermalDataset(
        storage, version=version, split=split, from_version=from_version
    )
    source_blobs = list_container_source_tiffs(storage)
    if not source_blobs:
        print(f"No TIFF files found under '{SOURCE_FILES_PREFIX}/' in '{container}'.")
        return

    summary = summarize_dataset_status(dataset, source_blobs)
    print_dataset_status(summary, show_all=False)
    if not summary.unlabelled_files:
        print("\nAll source files are already labelled or assigned as references.")
        return

    grouped = _group_source_blobs(source_blobs)
    for group, blobs in grouped:
        if group_filter and group_filter.lower() not in group.group_id.lower():
            continue

        unlabelled_in_group = [
            blob
            for blob in blobs
            if blob not in dataset.reference_tiff_files
            and blob not in dataset.labelled_sample_by_tiff
        ]
        if not unlabelled_in_group:
            continue

        print(
            f"\nGroup: {group.group_id} ({len(unlabelled_in_group)} unlabelled / {len(blobs)} total)"
        )
        reference = dataset.get_loaded_reference(group.group_id)
        if reference is None:
            reference, quit_requested = _pick_and_save_reference(
                dataset=dataset,
                group=group,
                candidate_blobs=unlabelled_in_group,
            )
            if quit_requested:
                print(
                    f"Saved {len(dataset.samples)} samples to {dataset.annotations_blob}"
                )
                return
            if reference is None:
                print(f"  No reference selected for {group.group_id}, skipping group.")
                continue

        for blob in unlabelled_in_group:
            if blob == reference.tiff_file:
                continue
            source_image = load_thermal_tiff_from_bytes(storage.read(blob))
            initial_polygon, auto_confidence = _estimate_initial_polygon(
                reference.image, source_image, reference.polygon
            )
            decision = label_thermal_sample(
                reference_image=reference.image,
                reference_polygon=reference.polygon,
                source_image=source_image,
                initial_polygon=initial_polygon,
                sample_label=PurePosixPath(blob).name,
                confidence=auto_confidence,
            )
            if decision.action == "quit":
                print(
                    f"Saved {len(dataset.samples)} samples to {dataset.annotations_blob}"
                )
                return
            if decision.action == "skip":
                continue

            sample_id = dataset.add_sample(
                reference_id=reference.reference_id,
                source_tiff_blob=blob,
                image=source_image,
                matchable=decision.action == "accept",
                gt_polygon=decision.gt_polygon,
            )
            print(f"  Saved sample #{sample_id} ({decision.action}) -> {blob}")

    final_summary = summarize_dataset_status(dataset, source_blobs)
    print()
    print_dataset_status(final_summary, show_all=False)


def validate_blob_dataset(
    version: str,
    split: str,
    storage_account: str = DEFAULT_STORAGE_ACCOUNT,
    container: str = DEFAULT_CONTAINER,
) -> ThermalSplit:
    storage = open_dataset_storage(storage_account, container)
    return validate_storage_dataset(storage, version=version, split=split)


def validate_storage_dataset(
    storage: BenchmarkStorage,
    version: str,
    split: str,
) -> ThermalSplit:
    thermal_split = ThermalSplit.load(storage.read, version=version, split=split)

    for ref_id, ref in thermal_split.references.items():
        thermal_split.read_reference_tiff(ref_id)
        _verify_preview_png(
            storage.read(ref.preview_file),
            ref.width,
            ref.height,
            f"reference {ref_id} preview",
        )

    for sample_id, sample in thermal_split.samples.items():
        thermal_split.read_source_tiff(sample_id)
        _verify_preview_png(
            storage.read(sample.preview_file),
            sample.width,
            sample.height,
            f"sample {sample_id} preview",
        )

    return thermal_split


def list_container_source_tiffs(storage: BenchmarkStorage) -> list[str]:
    return [
        name
        for name in storage.list_names(f"{SOURCE_FILES_PREFIX}/")
        if Path(name).suffix.lower() in (".tiff", ".tif")
    ]


def preview_path_for_source_blob(source_blob: str) -> str:
    prefix = f"{SOURCE_FILES_PREFIX}/"
    relative = (
        source_blob[len(prefix) :] if source_blob.startswith(prefix) else source_blob
    )
    rel_path = PurePosixPath(relative)
    png_path = rel_path.with_suffix(".png")
    return f"{PREVIEWS_PREFIX}/{png_path.as_posix()}"


def infer_group_for_source_blob(source_blob: str) -> SourceFileGroup:
    prefix = f"{SOURCE_FILES_PREFIX}/"
    relative = (
        source_blob[len(prefix) :] if source_blob.startswith(prefix) else source_blob
    )
    rel_path = PurePosixPath(relative)
    filename = rel_path.name
    if "__ThermalImage__" in filename:
        parts = filename.split("__")
        if len(parts) >= 3:
            tag_id = parts[0]
            inspection_description = parts[2].replace("-", " ")
            group_id = sanitize_identifier(f"{tag_id}__{inspection_description}")
            return SourceFileGroup(
                group_id=group_id,
                tag_id=tag_id,
                inspection_description=inspection_description,
            )

    if len(rel_path.parts) > 1:
        folder_name = rel_path.parts[0]
        return SourceFileGroup(
            group_id=sanitize_identifier(folder_name),
            tag_id=folder_name,
            inspection_description=folder_name,
        )

    return SourceFileGroup(
        group_id="default",
        tag_id="default",
        inspection_description="thermal",
    )


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(".-")
    return cleaned or "item"


def _group_source_blobs(
    source_blobs: list[str],
) -> list[tuple[SourceFileGroup, list[str]]]:
    groups_by_id: dict[str, SourceFileGroup] = {}
    blobs_by_id: dict[str, list[str]] = defaultdict(list)
    for blob in source_blobs:
        group = infer_group_for_source_blob(blob)
        groups_by_id[group.group_id] = group
        blobs_by_id[group.group_id].append(blob)
    return [
        (groups_by_id[group_id], sorted(blobs_by_id[group_id]))
        for group_id in sorted(groups_by_id)
    ]


def _pick_and_save_reference(
    dataset: BlobThermalDataset,
    group: SourceFileGroup,
    candidate_blobs: list[str],
) -> tuple[LoadedReference | None, bool]:
    for index, blob in enumerate(candidate_blobs):
        image = load_thermal_tiff_from_bytes(dataset.storage.read(blob))
        decision = draw_reference_polygon(
            candidate_image=image,
            candidate_label=f"{group.group_id} ({PurePosixPath(blob).name})",
            allow_next_candidate=index < len(candidate_blobs) - 1,
        )
        if decision.action == "quit":
            return None, True
        if decision.action == "accept" and decision.polygon is not None:
            loaded = dataset.add_reference(
                group=group,
                source_tiff_blob=blob,
                image=image,
                polygon=decision.polygon,
            )
            print(f"  Saved reference [{group.group_id}] -> {blob}")
            return loaded, False
    return None, False


def _resolve_blob_prefixes(folders_file: Path | None, prefix: str | None) -> list[str]:
    prefixes: list[str] = []
    if folders_file is not None:
        prefixes.extend(
            line.strip()
            for line in folders_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if prefix is not None:
        prefixes.append(prefix.strip())
    if not prefixes:
        raise ValueError("Provide --folders-file or --prefix")
    return prefixes


def _list_source_thermal_tiffs(
    source_store: BlobStore,
    prefixes: list[str],
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for folder in prefixes:
        normalized = folder.strip()
        for blob_name in source_store.list_blobs_by_prefix(prefix=normalized):
            if blob_name in seen:
                continue
            if (
                blob_name.lower().endswith((".tiff", ".tif"))
                and "__ThermalImage__" in blob_name
            ):
                seen.add(blob_name)
                found.append(blob_name)
    return sorted(found)


def _estimate_initial_polygon(
    reference_image: NDArray[np.float64],
    source_image: NDArray[np.float64],
    reference_polygon: Polygon,
) -> tuple[list[tuple[float, float]], float]:
    try:
        ref_uint8, src_uint8 = preprocess_thermal_pair_for_alignment(
            reference_image, source_image
        )
        alignment = align_thermal_images(
            ref_uint8, src_uint8, reference_polygon, method="phase_correlation"
        )
        return (
            [(float(x), float(y)) for x, y in alignment.warped_polygon],
            alignment.matching_confidence,
        )
    except Exception:
        return list(reference_polygon), 0.0


def _verify_preview_png(
    png_bytes: bytes,
    expected_width: int,
    expected_height: int,
    label: str,
) -> None:
    with Image.open(BytesIO(png_bytes)) as img:
        if img.size != (expected_width, expected_height):
            raise ValueError(
                f"Preview PNG size {img.size} does not match TIFF size "
                f"({expected_width}, {expected_height}) for {label}"
            )
