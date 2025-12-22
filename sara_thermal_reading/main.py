import argparse
import json
from typing import cast

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from numpy.typing import NDArray

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.file_io.blob import BlobStorageLocation
from sara_thermal_reading.logger import setup_logger

setup_logger()
from loguru import logger

from sara_thermal_reading.file_io.file_utils import (
    download_anonymized_image,
    load_reference_image_and_polygon,
    upload_to_visualized,
)
from sara_thermal_reading.image_alignment.align_two_images_orb_bf_cv2 import (
    align_two_images_orb_bf_cv2,
)
from sara_thermal_reading.visualization.create_annotated_thermal_visualization import (
    create_annotated_thermal_visualization,
)


def check_reference_blob_exists(
    tag_id: str, inspection_description: str, installation_code: str
) -> bool:
    logger.info(
        f"Checking if reference blob exists for tag_id: {tag_id}, inspection_description: {inspection_description}, installation_code: {installation_code}"
    )

    ref_blob_service_client = BlobServiceClient.from_connection_string(
        settings.REFERENCE_STORAGE_CONNECTION_STRING
    )
    img_path = f"{tag_id}_{inspection_description}/reference_image.jpeg"
    blob_client = ref_blob_service_client.get_blob_client(
        container=installation_code,
        blob=img_path,
    )

    exists = blob_client.exists()
    if exists:
        logger.info(f"Reference blob found at path: {img_path}")
    else:
        logger.warning(f"Reference blob not found at path: {img_path}")

    return exists


def find_max_temperature_in_polygon(
    thermal_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    temp_range: tuple[float, float] = (20.0, 100.0),
) -> tuple[float, tuple[int, int]]:
    """
    Find the maximum temperature within a polygon region of a thermal image.

    Maps pixel values to actual temperature values based on the thermal image's color scale.

    Args:
        thermal_image: The thermal image array (BGR format from OpenCV)
        polygon_points: Array of polygon vertices in format [[x1, y1], [x2, y2], ...]
        temp_range: Tuple of (min_temp, max_temp) representing the temperature scale of the image

    Returns:
        Tuple of (max_temperature_celsius, (x_coord, y_coord)) where the max temperature was found
    """
    # Convert polygon points to integer coordinates
    polygon_pts = polygon_points.reshape(-1, 2).astype(np.int32)

    # Create a mask for the polygon region
    height, width = thermal_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], (255,))

    # For thermal images, we need to extract temperature information more intelligently
    # Method 1: Try to use the thermal colormap information
    if len(thermal_image.shape) == 3:
        # Convert BGR to HSV for better thermal analysis
        hsv_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2HSV)

        # For thermal images, often the Value (brightness) or Hue channel contains temperature info
        # Use the Value channel as it typically correlates with temperature intensity
        thermal_intensity = hsv_thermal[:, :, 2]  # Value channel

        # Also try the original grayscale conversion as backup
        gray_thermal = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

        # Use the channel that shows more variation in the masked region
        masked_hsv_v = cv2.bitwise_and(thermal_intensity, thermal_intensity, mask=mask)
        masked_gray = cv2.bitwise_and(gray_thermal, gray_thermal, mask=mask)

        # Calculate standard deviation to see which channel has more information
        hsv_std = (
            np.std(masked_hsv_v[masked_hsv_v > 0]) if np.any(masked_hsv_v > 0) else 0
        )
        gray_std = (
            np.std(masked_gray[masked_gray > 0]) if np.any(masked_gray > 0) else 0
        )

        if hsv_std > gray_std:
            temperature_channel = thermal_intensity
            logger.info("Using HSV Value channel for temperature analysis")
        else:
            temperature_channel = gray_thermal
            logger.info("Using grayscale conversion for temperature analysis")
    else:
        temperature_channel = thermal_image
        logger.info("Using grayscale image for temperature analysis")

    # Apply mask to get only the polygon region
    masked_thermal = cv2.bitwise_and(
        temperature_channel, temperature_channel, mask=mask
    )

    # Find the maximum and minimum values in the masked region for calibration
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_thermal, mask=mask)

    # Also get global min/max from the entire image for reference
    global_min, global_max, _, _ = cv2.minMaxLoc(temperature_channel)

    logger.info(f"Pixel value range in polygon: {min_val} - {max_val}")
    logger.info(f"Global pixel value range: {global_min} - {global_max}")
    logger.info(f"Maximum pixel value location: {max_loc}")

    # Map pixel values to actual temperature
    # Method: Linear mapping from pixel range to temperature range
    min_temp, max_temp = temp_range

    # Use the global range for mapping to maintain consistency with the thermal scale
    if global_max > global_min:
        # Linear interpolation: temp = min_temp + (pixel_val - global_min) * (max_temp - min_temp) / (global_max - global_min)
        actual_temperature = min_temp + (max_val - global_min) * (
            max_temp - min_temp
        ) / (global_max - global_min)
    else:
        # Fallback if no variation detected
        actual_temperature = (min_temp + max_temp) / 2
        logger.warning("No temperature variation detected, using average temperature")

    # Clamp temperature to reasonable bounds
    actual_temperature = max(min_temp, min(max_temp, actual_temperature))

    logger.info(
        f"Mapped temperature: {actual_temperature:.1f}°C (from pixel value {max_val})"
    )

    return float(actual_temperature), cast(tuple[int, int], max_loc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get thermal reading inside polygon in image and upload visualization"
    )
    parser.add_argument(
        "--anonymizedBlobStorageLocation",
        required=True,
        help="JSON string for anonymized data blob storage location",
    )
    parser.add_argument(
        "--visualizedBlobStorageLocation",
        required=True,
        help="JSON string for visualized data blob storage location",
    )
    parser.add_argument(
        "--tagId",
        required=True,
        help="JSON string for is break output file",
    )
    parser.add_argument(
        "--inspectionDescription",
        required=True,
        help="JSON string for temperature output file",
    )
    parser.add_argument(
        "--installationCode",
        required=True,
        help="JSON string for installation code",
    )
    parser.add_argument(
        "--temperature-output-file",
        required=False,
        help="JSON string for temperature output file",
        default="/tmp/temperature_output.txt",
    )

    args = parser.parse_args()
    print(f"Arguments received: {args}")
    try:
        anonymized_blob_storage_location = BlobStorageLocation.model_validate(
            json.loads(args.anonymizedBlobStorageLocation)
        )
        visualized_blob_storage_location = BlobStorageLocation.model_validate(
            json.loads(args.visualizedBlobStorageLocation)
        )
        tag_id: str = args.tagId
        inspection_description: str = args.inspectionDescription
        installation_code: str = args.installationCode
        temperature_output_file: str = args.temperature_output_file

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON provided: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing input: {e}")
    if not check_reference_blob_exists(
        tag_id, inspection_description, installation_code
    ):
        logger.error(
            f"Expecting reference image to exist on storage account {settings.REFERENCE_STORAGE_ACCOUNT} for tagId {tag_id} and inspectionDescription {inspection_description} on installationCode {installation_code}"
        )
        return

    reference_image, reference_polygon = load_reference_image_and_polygon(
        installation_code, tag_id, inspection_description
    )

    # Download the source image
    source_image_array = download_anonymized_image(anonymized_blob_storage_location)

    warped_polygon, aligned_image = align_two_images_orb_bf_cv2(
        reference_image,
        source_image_array,
        reference_polygon,
    )

    # Find the maximum temperature within the polygon region
    max_temperature, max_temp_location = find_max_temperature_in_polygon(
        aligned_image, warped_polygon
    )

    logger.info(
        f"Maximum temperature found: {max_temperature} at location {max_temp_location}"
    )

    # Create annotated visualization
    annotated_image = create_annotated_thermal_visualization(
        aligned_image,
        warped_polygon,
        max_temperature,
        max_temp_location,
        tag_id,
        inspection_description,
    )

    logger.info(f"Created annotated thermal visualization")

    upload_to_visualized(
        visualized_blob_storage_location,
        annotated_image,
    )

    with open(temperature_output_file, "w") as file:
        file.write(str(max_temperature))
        print(
            f"Max temperature: {max_temperature} written to {temperature_output_file}"
        )


if __name__ == "__main__":
    main()
