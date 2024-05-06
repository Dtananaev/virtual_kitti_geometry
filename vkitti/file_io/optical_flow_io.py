#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import cv2
import numpy as np


def read_vkitti_png_flow(flow_filename: str)-> np.ndarray:
  """Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"""
  # read png to bgr in 16 bit unsigned short

  bgr = np.asarray(cv2.imread(flow_filename,  -1), dtype=np.float32)
  h, w, _ = bgr.shape
  # b == invalid flow flag == 0 for sky or other invalid flow
  invalid = bgr[..., 0] == 0
  # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
  out_flow = 2.0 / (2**16 -1) * bgr[..., 2:0:-1] - 1
  out_flow[..., 0] *=w-1
  out_flow[..., 1] *=h-1
  out_flow[invalid] = 0 # or another value (e.g., np.nan)
  return np.asarray(out_flow, dtype=np.float32), invalid