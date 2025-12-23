import random
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def random_translate_thermal_img(
    image: NDArray[np.float64],
    min_percent: float = 0.05,
    max_percent: float = 0.1,
) -> NDArray[np.float64]:
    """
    Randomly translates a thermal image in a random direction.
    Fills the new blank space with the mean value of the image.

    Args:
        image: Input thermal image (float64).
        min_percent: Minimum translation distance as a percentage of the image size (0.0-1.0).
        max_percent: Maximum translation distance as a percentage of the image size (0.0-1.0).

    Returns:
        Translated thermal image.
    """
    height, width = image.shape[:2]
    mean_value = float(np.mean(image))

    image_size = max(height, width)
    distance_percent = random.uniform(min_percent, max_percent)
    distance_pixels = distance_percent * image_size

    angle = random.uniform(0, 2 * np.pi)

    tx = distance_pixels * np.cos(angle)
    ty = distance_pixels * np.sin(angle)

    affine_transformation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    translated_image = cv2.warpAffine(
        image,
        affine_transformation_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(mean_value,),
    )

    return cast(NDArray[np.float64], translated_image)
