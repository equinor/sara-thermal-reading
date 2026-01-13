import random

import cv2
import numpy as np
from numpy.typing import NDArray

from sara_thermal_reading.dev_utils.image_warping import random_translate_thermal_img


def _create_noisy_background(height: int, width: int) -> NDArray[np.uint8]:
    ref_image = np.zeros((height, width), dtype=np.uint8) + 30

    noise = np.random.normal(0, 10, (height, width)).astype(np.int16)
    ref_image = np.clip(ref_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return ref_image


def _add_distractor_shapes(image: NDArray[np.uint8], num_distractors: int) -> None:
    for _ in range(num_distractors):
        shape_type = random.choice(["circle", "rect", "triangle"])
        x, y = random.randint(50, 450), random.randint(50, 450)
        color = random.randint(80, 200)

        if shape_type == "circle":
            radius = random.randint(20, 50)
            cv2.circle(image, (x, y), radius, (color,), -1)
        elif shape_type == "rect":
            w, h = random.randint(30, 80), random.randint(30, 80)
            cv2.rectangle(image, (x, y), (x + w, y + h), (color,), -1)
        elif shape_type == "triangle":
            size = random.randint(30, 80)
            pts = np.array(
                [
                    [x, y - size // 2],
                    [x - size // 2, y + size // 2],
                    [x + size // 2, y + size // 2],
                ],
                np.int32,
            )
            cv2.fillPoly(image, [pts], (color,))  # type: ignore


def _create_diamond_polygon(cx: int, cy: int, size: int) -> list[tuple[int, int]]:
    return [
        (cx, cy - size),
        (cx + size, cy),
        (cx, cy + size),
        (cx - size, cy),
    ]


def generate_synthetic_alignment_data(
    seed: int,
) -> tuple[
    NDArray[np.uint8], NDArray[np.uint8], list[tuple[int, int]], tuple[float, float]
]:
    """Generates synthetic thermal data for testing alignment algorithms.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        tuple: (reference_image, source_image, roi_polygon, (tx, ty))
    """
    np.random.seed(seed)
    random.seed(seed)

    height, width = 500, 500

    ref_image = _create_noisy_background(height, width)

    _add_distractor_shapes(ref_image, num_distractors=5)

    cx, cy = 250, 250
    poly_size = 60
    roi_polygon = _create_diamond_polygon(cx, cy, poly_size)

    pts = np.array(roi_polygon, np.int32)
    cv2.fillPoly(ref_image, [pts], (250,))  # type: ignore

    ref_float = ref_image.astype(np.float64)
    translated_float, translation = random_translate_thermal_img(ref_float)

    source_image = np.clip(translated_float, 0, 255).astype(np.uint8)

    return ref_image, source_image, roi_polygon, translation
