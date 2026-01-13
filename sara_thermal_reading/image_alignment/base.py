from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ImageAlignmentStrategy(ABC):
    @abstractmethod
    def align(
        self,
        reference_image: NDArray[np.uint8],
        source_image: NDArray[np.uint8],
        roi_polygon: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Aligns the reference image context to the source image and warps the polygon.

        Args:
            reference_image: The reference thermal image as a numpy array (uint8).
            source_image: The source (current) thermal image as a numpy array (uint8).
            roi_polygon: A list of (x, y) tuples representing the polygon on the
                reference image.

        Returns:
            list[tuple[int, int]]: The coordinates of the polygon warped to match
                the source image.
        """
        pass
