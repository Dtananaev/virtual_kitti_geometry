#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import os
import numpy as np
import glob
import cv2
from vkitti.statics import intrinsic_dict
from vkitti.file_io.optical_flow_io import read_vkitti_png_flow
from vkitti.visualization.depth_visu import depth_to_image, disp2rgb
from vkitti.visualization.optical_flow_visu import flow_to_image, point_vec
from vkitti.tensor_tools.warp import inverse_warp_from_flow
from vkitti.visualization.put_text import add_legend



def compute_background_mask(mid: np.ndarray, alpha: float=0.1)-> np.ndarray:
    """Compute background mask."""
    
    height, width = mid.shape
    gradient_y = mid[:-1, :] - mid[1:, :]
    new_row = np.zeros((1, width), dtype=np.float32) 
    gradient_y = abs(np.vstack((gradient_y, new_row)))



    mid_grad_y  = gradient_y * mid

    # Define the 3x1 kernel for averaging
    kernel = np.ones((5, 1), np.float32) / 5

    # Apply convolution to calculate the average of each set of three pixels in each column
    averages = cv2.filter2D(mid_grad_y, -1, kernel)

    background_mask  = mid_grad_y > alpha*averages


    return background_mask




def visualize_dataset(dataset_dir: str, scene: str, output_dir: str)-> None:
    """Visualize dataset."""
    
    visu_output_dir = os.path.join(output_dir, scene)
    os.makedirs(visu_output_dir, exist_ok=True)
    
    rgb_path = os.path.join(dataset_dir, scene, "clone", "frames", "rgb", "Camera_0")
    depth_path = os.path.join(dataset_dir, scene, "clone", "frames", "depth", "Camera_0")
    forward_flow_path = os.path.join(dataset_dir, scene, "clone", "frames", "forwardFlow", "Camera_0")


    rgb_list = sorted(glob.glob(rgb_path + "/*.jpg"))

    #intrinsic = intrinsic_dict[scene]

    for idx, rgb_file in enumerate(rgb_list):
        
        rgb_basename = os.path.basename(rgb_file)
        rgb_basename_string = f"rgb_{idx:05d}.jpg"
        assert rgb_basename == rgb_basename_string


        image = np.asarray(cv2.imread(rgb_file), dtype=np.float32)
        depth_filename= os.path.join(depth_path, rgb_basename.replace("rgb_", "depth_").replace(".jpg", ".png"))
        # Depth in meters
        depth = np.asarray(cv2.imread(depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), dtype=np.float32)/ 100.0

        flow_filename = os.path.join(forward_flow_path, rgb_basename.replace("rgb_", "flow_").replace(".jpg", ".png"))
        flow, invalid = read_vkitti_png_flow(flow_filename)


        # Compute motion in depth
        if idx < len(rgb_list):

            next_depth_filename = os.path.join(depth_path, f"depth_{idx+1:05d}.png")
            depth2 = np.asarray(cv2.imread(next_depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), dtype=np.float32)/ 100.0
            depth2_aligned = inverse_warp_from_flow(depth2, flow)
            invalid_mask  = invalid | (depth2_aligned == 0.0)

            mid = depth2_aligned / depth
            mid[invalid_mask] = 1.0
            mid_visu = np.clip(mid, 0.5, 1.5)
            height, width = mid_visu.shape

            background_mask = compute_background_mask(mid=mid, alpha=0.1)

          
          



            norm_grad_y_img=  np.zeros_like(image)
            norm_grad_y_img[background_mask] = 255.0


            # For visu
            mid_visu = (mid_visu - 0.5) / 1.0
            mid_image = disp2rgb(mid_visu)
            mid_image[invalid_mask] = 0.0
        else:
            mid_image = np.zeros_like(image)
            norm_grad_y_img=  np.zeros_like(image)
        mid_image = add_legend(mid_image, legend="motion in depth")
        norm_grad_y_img = add_legend(norm_grad_y_img, legend="gradient y in motion in depth")

        depth_visu = np.clip(depth, 0.0, 100.0) 
        depth_image = depth_to_image(image=image, depth=depth_visu, alpha = 0.4)
        depth_image = add_legend(depth_image, legend="depth")
        flow_image = point_vec(img=image, flow=flow, skip=20)
        flow_image = add_legend(flow_image, legend="optical flow")


        total_visu_top = np.hstack((flow_image, depth_image))
        total_visu_bottom = np.hstack((mid_image, norm_grad_y_img))
        
        total_visu = np.vstack((total_visu_top,total_visu_bottom))
        visu_filename_to_save = os.path.join(visu_output_dir, rgb_basename)
        cv2.imwrite(visu_filename_to_save, total_visu[..., ::-1])

def get_arguments():
    parser = argparse.ArgumentParser(description="virtual kitti visu dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/media/denis/SSD_A/virtual_kitti",
        help="vkitti dataset dir.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="Scene18",
        help="vkitti dataset dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/denis/SSD_A/virtual_kitti_output_visu",
        help="vkitti dataset dir.",
    )
    args = parser.parse_args()
    return args





if __name__ == '__main__':
    args = get_arguments()
    visualize_dataset(dataset_dir=args.dataset_dir, scene=args.scene, output_dir=args.output_dir)






