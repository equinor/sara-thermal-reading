import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.image_alignment.align_dispatcher import (
    AlignmentMethod,
    align_thermal_images,
    phase_correlation_to_matching_confidence,
)
from sara_thermal_reading.image_processing.convert_thermal_to_uint8 import (
    convert_thermal_to_uint8,
)
from sara_thermal_reading.image_processing.find_temperature_in_polygon import (
    find_temperature_in_polygon,
)
from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BlobStorageLocation",
    "ThermalReadingResult",
    "evaluate_thermal_reading",
    "phase_correlation_to_matching_confidence",
    "process_thermal_image",
    "run_thermal_reading_workflow",
]


@dataclass(frozen=True)
class ThermalReadingResult:
    temperature: float
    warped_polygon: list[tuple[int, int]]
    warped_reference_image: NDArray[np.uint8]
    raw_score: float
    matching_confidence: float


def preprocess_thermal_pair_for_alignment(
    reference_image: NDArray[np.float64],
    source_image: NDArray[np.float64],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    shared_vmin = max(float(np.min(reference_image)), float(np.min(source_image)))
    shared_vmax = min(float(np.max(reference_image)), float(np.max(source_image)))
    shared_clip_range = (shared_vmin, shared_vmax)

    reference_uint8 = convert_thermal_to_uint8(
        reference_image, clip_range=shared_clip_range
    )
    source_uint8 = convert_thermal_to_uint8(source_image, clip_range=shared_clip_range)

    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    reference_uint8 = clahe.apply(reference_uint8).astype(np.uint8)
    source_uint8 = clahe.apply(source_uint8).astype(np.uint8)
    return reference_uint8, source_uint8


def evaluate_thermal_reading(
    reference_image: NDArray[np.float64],
    source_image_array: NDArray[np.float64],
    reference_polygon: Sequence[tuple[float, float]],
    alignment_method: AlignmentMethod = "phase_correlation",
) -> ThermalReadingResult:
    reference_uint8, source_uint8 = preprocess_thermal_pair_for_alignment(
        reference_image, source_image_array
    )
    alignment = align_thermal_images(
        reference_uint8,
        source_uint8,
        reference_polygon,
        method=alignment_method,
    )
    if alignment.matching_confidence < 1.0:
        logger.warning(
            f"Low alignment score ({alignment.raw_score:.3f}) for method {alignment_method}, "
            f"maps to confidence score: {alignment.matching_confidence:.2f}."
        )

    warped_polygon_array = np.array(alignment.warped_polygon)
    temperature = find_temperature_in_polygon(
        source_image_array, warped_polygon_array, settings.TEMPERATURE_PERCENTILE
    )
    logger.info(
        f"Temperature inside the polygon (P{settings.TEMPERATURE_PERCENTILE:g}): {temperature}."
    )
    return ThermalReadingResult(
        temperature=temperature,
        warped_polygon=alignment.warped_polygon,
        warped_reference_image=alignment.warped_reference_image,
        raw_score=alignment.raw_score,
        matching_confidence=alignment.matching_confidence,
    )


def process_thermal_image(
    reference_image: NDArray[np.float64],
    source_image_array: NDArray[np.float64],
    reference_polygon: Sequence[tuple[float, float]],
    alignment_method: AlignmentMethod = "phase_correlation",
) -> tuple[
    float,
    NDArray[np.uint8],
    list[tuple[int, int]],
    NDArray[np.uint8],
    float,
]:
    result = evaluate_thermal_reading(
        reference_image=reference_image,
        source_image_array=source_image_array,
        reference_polygon=reference_polygon,
        alignment_method=alignment_method,
    )
    warped_polygon_array = np.array(result.warped_polygon, dtype=np.float32)
    annotated_image = create_annotated_thermal_visualization(
        source_image_array,
        warped_polygon_array,
        result.temperature,
        settings.TEMPERATURE_PERCENTILE,
    )
    return (
        result.temperature,
        annotated_image,
        result.warped_polygon,
        result.warped_reference_image,
        result.matching_confidence,
    )


def run_thermal_reading_workflow(
    anonymized_blob_storage_location: BlobStorageLocation,
    visualized_blob_storage_location: BlobStorageLocation,
    reference_image_blob_storage_location: BlobStorageLocation,
    reference_polygon: Sequence[tuple[float, float]],
    result_output_file: str,
) -> None:
    logger.info("Starting run thermal reading workflow")
    logger.info(
        "Loading reference image with %d supplied polygon vertices",
        len(reference_polygon),
    )
    reference_image_blob_store = BlobStore(
        installation_code=reference_image_blob_storage_location.blob_container,
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    if not reference_image_blob_store.check_if_exists(
        reference_image_blob_storage_location.blob_name
    ):
        logger.error(
            "Reference tiff image does not exist on %s/%s",
            reference_image_blob_storage_location.blob_container,
            reference_image_blob_storage_location.blob_name,
        )
        raise Exception(
            f"Reference tiff image does not exist on {reference_image_blob_storage_location.blob_container}/{reference_image_blob_storage_location.blob_name}"
        )
    reference_image: np.ndarray = reference_image_blob_store.download_thermal_tiff(
        reference_image_blob_storage_location.blob_name
    )

    logger.info("Downloaded reference image")
    logger.info("Downloading thermal TIFF image from anonymized")
    anonymized_blob_store = BlobStore(
        installation_code=anonymized_blob_storage_location.blob_container,
        connection_string=settings.SOURCE_STORAGE_CONNECTION_STRING,
    )
    source_image: np.ndarray = anonymized_blob_store.download_thermal_tiff(
        blob_name=anonymized_blob_storage_location.blob_name
    )
    logger.info(
        f"Downloaded TIFF image from source storage account, shape: {source_image.shape}"
    )

    logger.info("Processing thermal image")
    temperature, annotated_image, _, _, matching_confidence = process_thermal_image(
        reference_image, source_image, reference_polygon
    )
    logger.info("Created annotated thermal visualization")

    logger.info("Uploading to visualized")
    visualized_blob_store = BlobStore(
        installation_code=visualized_blob_storage_location.blob_container,
        connection_string=settings.DESTINATION_STORAGE_CONNECTION_STRING,
    )
    visualized_blob_store.upload_jpg(
        annotated_image, blob_name=visualized_blob_storage_location.blob_name
    )
    logger.info(
        f"Uploaded annotated thermal image to visualized storage account: {visualized_blob_storage_location.blob_container}/{visualized_blob_storage_location.blob_name}"
    )

    with open(result_output_file, "w") as file:
        result = {
            "temperature": float(temperature),
            "confidence": float(matching_confidence),
            "outputBlobStorageLocation": {
                "storageAccount": visualized_blob_storage_location.storage_account,
                "blobContainer": visualized_blob_storage_location.blob_container,
                "blobName": visualized_blob_storage_location.blob_name,
            },
        }
        json.dump(result, file)
        logger.info(f"Temperature: {temperature} written to {result_output_file}")
