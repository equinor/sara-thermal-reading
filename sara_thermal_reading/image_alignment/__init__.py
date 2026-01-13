from sara_thermal_reading.config.settings import ImageAlignmentMethod

from .align_two_images_orb_bf_cv2 import OrbBfCv2Aligner
from .base import ImageAlignmentStrategy
from .warp_polygon_orb_bf_cv2 import WarpPolygonAligner


def get_alignment_strategy(method: ImageAlignmentMethod) -> ImageAlignmentStrategy:
    if method == ImageAlignmentMethod.ORB_BF_CV2:
        return OrbBfCv2Aligner()
    elif method == ImageAlignmentMethod.WARP_POLYGON:
        return WarpPolygonAligner()
    else:
        raise ValueError(f"Unknown alignment method: {method}")
