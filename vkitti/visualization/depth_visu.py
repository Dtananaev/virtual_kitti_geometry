# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def disp2rgb(disp):
    H = disp.shape[0]
    W = disp.shape[1]

    I = disp.flatten()

    map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]])
    bins = map[:-1,3]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]

    ind = np.minimum(np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None],
                                    I.shape[0], axis=1), axis=0), 6)
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:, None])

    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(np.maximum(np.multiply(map[ind,0:3], np.repeat(1-I[:,None], 3, axis=1)) \
         + np.multiply(map[ind+1,0:3], np.repeat(I[:,None], 3, axis=1)),0),1)

    I = np.reshape(I, [H, W, 3]).astype(np.float32)

    return 255* I

def depth_to_image(image: np.ndarray, depth: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Creates blended image with depth map.

    Args:
       image: input image of shape [ height, width, 3]
       depth: depth batch of shape [height, width]

    Returns:
        tinted_image: blended depth image [height, width, channels]
    """
    tinted_images = []
    cmap = plt.cm.turbo

    depth_relative = depth / (np.percentile(depth, 95) + 1e-8)
    d_image = 255.0 * cmap(np.squeeze(np.clip(depth_relative, 0.0, 1.0)))[..., :3]
    
    tinted_img = alpha * image + (1.0 - alpha) * d_image
    return np.asarray(tinted_img, dtype=np.float32)