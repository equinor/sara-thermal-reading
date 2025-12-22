import cv2
import numpy as np
from numpy.typing import NDArray


def create_annotated_thermal_visualization(
    aligned_image: NDArray[np.uint8],
    polygon_points: NDArray[np.float32],
    max_temperature: float,
    max_temp_location: tuple[int, int],
    tag_id: str,
    inspection_description: str,
) -> NDArray[np.uint8]:
    """
    Create an annotated visualization of the aligned thermal image with polygon and temperature info.

    Args:
        aligned_image: The aligned thermal image
        polygon_points: Array of polygon vertices
        max_temperature: The maximum temperature value found
        max_temp_location: (x, y) coordinates of the maximum temperature
        tag_id: Tag identifier for the image
        inspection_description: Description of the inspection

    Returns:
        The annotated image array
    """
    # Create a copy of the aligned image for annotation
    annotated_image = aligned_image.copy()

    # Convert polygon points to integer coordinates
    polygon_pts = polygon_points.reshape(-1, 2).astype(np.int32)

    # Draw the polygon outline in bright green
    cv2.polylines(annotated_image, [polygon_pts], True, (0, 255, 0), 3)

    # Draw a circle at the max temperature location in bright red
    cv2.circle(annotated_image, max_temp_location, 8, (0, 0, 255), -1)

    # Add temperature label next to the marker
    temp_label = f"{max_temperature:.1f}"
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.7
    label_thickness = 2

    # Calculate label position (offset from the marker)
    label_x = max_temp_location[0] + 15
    label_y = max_temp_location[1] - 10

    # Get text size for background rectangle
    (label_width, label_height), label_baseline = cv2.getTextSize(
        temp_label, label_font, label_font_scale, label_thickness
    )

    # Draw semi-transparent background for the temperature label
    overlay = annotated_image.copy()
    cv2.rectangle(
        overlay,
        (label_x - 3, label_y - label_height - 3),
        (label_x + label_width + 6, label_y + label_baseline + 3),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

    # Draw the temperature label in bright yellow for high visibility
    cv2.putText(
        annotated_image,
        temp_label,
        (label_x, label_y),
        label_font,
        label_font_scale,
        (0, 255, 255),
        label_thickness,
    )

    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Prepare text information
    temp_text = f"Max Temp: {max_temperature:.1f}"
    location_text = f"Location: ({max_temp_location[0]}, {max_temp_location[1]})"
    tag_text = f"Tag: {tag_id}"
    inspection_text = f"Inspection: {inspection_description}"

    # Calculate text positions (top-left area)
    y_offset = 30
    x_margin = 10

    # Add background rectangles for better text visibility
    texts = [tag_text, inspection_text, temp_text, location_text]

    for i, text in enumerate(texts):
        y_pos = y_offset + i * 35

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw semi-transparent background rectangle
        overlay = annotated_image.copy()
        cv2.rectangle(
            overlay,
            (x_margin - 5, y_pos - text_height - 5),
            (x_margin + text_width + 10, y_pos + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

        # Draw the text in white
        cv2.putText(
            annotated_image,
            text,
            (x_margin, y_pos),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # Add a legend for the colors
    legend_y = annotated_image.shape[0] - 80
    cv2.putText(
        annotated_image, "Legend:", (x_margin, legend_y), font, 0.6, (255, 255, 255), 2
    )
    cv2.putText(
        annotated_image,
        "Green: ROI Polygon",
        (x_margin, legend_y + 25),
        font,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_image,
        "Red: Max Temperature",
        (x_margin, legend_y + 45),
        font,
        0.5,
        (0, 0, 255),
        2,
    )

    return annotated_image
