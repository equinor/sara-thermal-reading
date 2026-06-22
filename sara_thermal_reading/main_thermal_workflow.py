import json
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.models.blob_storage_location import BlobStorageLocation

logger = logging.getLogger(__name__)

from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.image_alignment.align_two_images_translation_cv2 import (
    align_two_images_translation_cv2,
)
from sara_thermal_reading.image_processing.convert_thermal_to_uint8 import (
    convert_thermal_to_uint8,
)
from sara_thermal_reading.image_processing.find_temperature_in_polygon import (
    find_temperature_in_polygon,
)
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)

__all__ = [
    "BlobStorageLocation",
    "process_thermal_image",
    "run_thermal_reading_workflow",
]


def process_thermal_image(
    reference_image: NDArray[np.float64],
    source_image_array: NDArray[np.float64],
    reference_polygon: list[tuple[int, int]],
) -> tuple[
    float,
    NDArray[np.uint8],
    list[tuple[int, int]],
    NDArray[np.uint8],
    float,
]:

    # Normalize both images to their shared (overlapping) temperature range so
    # that a given temperature maps to the same brightness in both images.
    # This prevents absolute temperature differences between captures from
    # affecting the matching.
    shared_vmin = max(float(np.min(reference_image)), float(np.min(source_image_array)))
    shared_vmax = min(float(np.max(reference_image)), float(np.max(source_image_array)))
    shared_clip_range = (shared_vmin, shared_vmax)

    reference_image_uint8 = convert_thermal_to_uint8(
        reference_image, clip_range=shared_clip_range
    )
    source_image_uint8 = convert_thermal_to_uint8(
        source_image_array, clip_range=shared_clip_range
    )

    # Apply CLAHE to enhance local structural contrast (edges, gradients) in both
    # images equally. ORB keypoints are gradient-based, so this makes feature
    # detection focus on structure rather than absolute brightness levels.
    clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
    reference_image_uint8 = clahe.apply(reference_image_uint8).astype(np.uint8)
    source_image_uint8 = clahe.apply(source_image_uint8).astype(np.uint8)

    warped_polygon_list, warped_reference_img, phase_correlation = (
        align_two_images_translation_cv2(
            reference_image_uint8,
            source_image_uint8,
            reference_polygon,
        )
    )
    warped_polygon_array = np.array(warped_polygon_list)
    matching_confidence = phase_correlation_to_matching_confidence(phase_correlation)
    if matching_confidence < 1.0:
        logger.warning(
            f"Low phase correlation response ({phase_correlation:.3f}), translation estimate may be unreliable. Maps to confidence score: {matching_confidence:.2f}. "
        )

    temperature = find_temperature_in_polygon(
        source_image_array, warped_polygon_array, settings.TEMPERATURE_PERCENTILE
    )

    logger.info(
        f"Temperature inside the polygon (P{settings.TEMPERATURE_PERCENTILE:g}): {temperature}."
    )

    annotated_image = create_annotated_thermal_visualization(
        source_image_array,
        warped_polygon_array,
        temperature,
        settings.TEMPERATURE_PERCENTILE,
    )

    return (
        temperature,
        annotated_image,
        warped_polygon_list,
        warped_reference_img,
        matching_confidence,
    )


def phase_correlation_to_matching_confidence(phase_correlation: float) -> float:
    """
    Convert phase correlation response to a confidence score between 0 and 1.
    This is a heuristic mapping based on expected ranges of phase correlation values.
    """
    if phase_correlation < settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION:
        return 0.0
    elif phase_correlation > settings.CONFIDENCE_CALC_LINEAR_MAX_PHASE_CORRELATION:
        return 1.0
    else:
        # Linearly interpolate between min and max scores
        return (
            phase_correlation - settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION
        ) / (
            settings.CONFIDENCE_CALC_LINEAR_MAX_PHASE_CORRELATION
            - settings.CONFIDENCE_CALC_LINEAR_MIN_PHASE_CORRELATION
        )


def run_thermal_reading_workflow(
    anonymized_blob_storage_location: BlobStorageLocation,
    visualized_blob_storage_location: BlobStorageLocation,
    reference_image_blob_storage_location: BlobStorageLocation,
    reference_polygon_blob_storage_location: BlobStorageLocation,
    result_output_file: str,
) -> None:

    logger.info(f"Starting run thermal reading workflow")

    logger.info(f"Loading reference image and polygon")
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

    reference_polygon_blob_store = BlobStore(
        installation_code=reference_polygon_blob_storage_location.blob_container,
        connection_string=settings.REFERENCE_STORAGE_CONNECTION_STRING,
    )
    if not reference_polygon_blob_store.check_if_exists(
        reference_polygon_blob_storage_location.blob_name
    ):
        logger.error(
            "Reference polygon does not exist on %s/%s",
            reference_polygon_blob_storage_location.blob_container,
            reference_polygon_blob_storage_location.blob_name,
        )
        raise Exception(
            f"Reference polygon does not exist on {reference_polygon_blob_storage_location.blob_container}/{reference_polygon_blob_storage_location.blob_name}"
        )
    reference_polygon: list[tuple[int, int]] = (
        reference_polygon_blob_store.download_polygon(
            reference_polygon_blob_storage_location.blob_name
        )
    )
    logger.info(f"Downloaded reference image and polygon")

    logger.info(f"Downloading thermal TIFF image from anonymized")
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

    logger.info(f"Processing thermal image")
    temperature, annotated_image, _, _, matching_confidence = process_thermal_image(
        reference_image, source_image, reference_polygon
    )
    logger.info(f"Created annotated thermal visualization")

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
        json.dump(
            {
                "temperature": float(temperature),
                "confidence": float(matching_confidence),
            },
            file,
        )
        logger.info(f"Temperature: {temperature} written to {result_output_file}")
