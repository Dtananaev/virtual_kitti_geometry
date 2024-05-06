#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import cv2


def add_legend(image, legend, color_legend=[255, 255, 255], color_background=[0, 0, 0]):
    """Puts text in image."""
    image = image.copy()
    height, width,  _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =1
    thickness_text = 2
    text_size, _ = cv2.getTextSize(legend, font, font_scale, thickness_text)
    text_w, text_h  = text_size
    center_x = width / 2

    x0 = int(center_x - text_w / 2.0)
    y0 = int(height - text_h - 5)
    x1 = int(center_x + text_w / 2.0)
    y1 = int(height -5)

    cv2.rectangle(image, (x0, y0), (x1, y1), color=color_background, thickness=-1)
    cv2.putText(image, legend, org=(x0, y1), fontFace=font, fontScale=font_scale, color=color_legend, thickness=thickness_text

    )
    return image