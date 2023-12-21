from typing import *
import itertools
import numpy as np

from scipy.optimize import linear_sum_assignment


def mask_to_bbox(mask: np.ndarray, min_size) -> np.ndarray:
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for idx, ax in enumerate(itertools.combinations(range(mask.ndim), mask.ndim - 1)):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0

        # Uncomment for MRI data
        if right - left < min_size[2-idx]:
            offset = 128 - (right - left)

            if (left - offset//2) >= 0 or (right + offset//2 + offset%2) < mask.shape[2-idx]:
                right += offset//2 + offset%2
                left -= offset//2
            elif (left - offset//2) < 0:
                right += offset - left
                left = 0
            elif (right + offset//2 + offset%2) >= mask.shape[2-idx]:
                left -= offset - (mask.shape[2-idx] - 1 - right)
                right = mask.shape[2-idx] - 1

        start.insert(0, left)
        stop.insert(0, right + 1)

    return np.array([start, stop])


def limit_box(box: np.ndarray, limit: Union[int, Sequence[int]]) -> np.ndarray:
    start, stop = box
    start = np.maximum(start, 0)
    stop = np.minimum(stop, limit)
    return np.array([start, stop])
