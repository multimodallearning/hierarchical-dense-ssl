import numpy as np
from imops import pad
from skimage import morphology
from skimage.segmentation import flood



BODY_THRESHOLD_HU = -500
BODY_THRESHOLD_MRI = 0.03


def get_body_mask(image: np.ndarray, threshold: float) -> np.ndarray:

    air = image < threshold
    body_mask = ~flood(pad(air, padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0, 0))[1:-1, 1:-1]

    # Uncomment for MRI data
    # body_mask = np.zeros_like(air)
    # for slice_idx in range(air.shape[2]):
    #     flood_slice = ~flood(pad(air[..., slice_idx], padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0))[1:-1, 1:-1]
    #     body_mask[..., slice_idx] = morphology.remove_small_objects(flood_slice, min_size=1000)

    return body_mask
