import cv2
import numpy as np
import torch

from fiontb.fiontblib import filter_depth_image

def blur_depth_image1(depth_image, kernel_size, fg_mask):
    kernel = np.ones((kernel_size, kernel_size))

    sum_image = cv2.filter2D(depth_image, cv2.CV_16U, kernel)
    count = cv2.filter2D(fg_mask.astype(np.uint8), cv2.CV_16U, kernel)
    # non_zero = fg_mask > 0
    non_zero = count == kernel_size*kernel_size

    depth_image[non_zero] = sum_image[non_zero] / count[non_zero]
    return depth_image

def blur_depth_image(depth_image, kernel_size, mask):
    kernel = torch.ones((kernel_size, kernel_size), dtype=torch.int16)

    depth_image = torch.from_numpy(depth_image.astype(np.int16))
    mask = torch.from_numpy(mask).byte()

    depth_image = filter_depth_image(depth_image, mask, kernel)
    return depth_image.numpy()
