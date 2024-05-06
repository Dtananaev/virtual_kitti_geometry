# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import numpy as np
import cv2


def inverse_warp_from_flow(tensor:np.ndarray, optical_flow:np.ndarray)->np.ndarray:
    """Warp tensor from flow."""

    height, width, _ = optical_flow.shape
    xx, yy = np.meshgrid(range(width), range(height), indexing='xy')
    coords = np.asarray(np.stack((xx, yy), axis=-1), dtype=np.float32)
    maptable = coords + optical_flow
    warped_tensor = cv2.remap(tensor, maptable[..., 0], maptable[..., 1], cv2.INTER_NEAREST)

    return warped_tensor

