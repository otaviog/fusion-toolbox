"""General testing facilitators for intern fusion toolbox tests.

"""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np


from fiontb.processing import (estimate_normals, bilateral_depth_filter,
                               ColorSpace, to_color_feature)
from fiontb.data.ftb import load_ftb



def preprocess_frame(frame, scale=1, filter_depth=True, color_space=ColorSpace.LAB,
                     blur=False, compute_normals=False):
    """Preprocess a frame and extract color features.
    """

    height, width = frame.depth_image.shape
    height, width = int(height*scale), int(width*scale)

    if scale < 1:
        frame.depth_image = cv2.resize(frame.depth_image.astype(
            np.uint16), (width, height)).astype(np.int32)
        frame.rgb_image = cv2.resize(frame.rgb_image,
                                     (width, height))
        frame.info.kcam = frame.info.kcam.scaled(scale)

    mask = frame.depth_image > 0
    if filter_depth:
        frame.depth_image = bilateral_depth_filter(
            frame.depth_image,
            mask).numpy()

    if compute_normals:
        normal_depth_image = frame.depth_image
        if not filter_depth:
            normal_depth_image = bilateral_depth_filter(
                frame.depth_image.to("cuda:0"),
                mask.to("cuda:0")).numpy()

        frame.normal_image = estimate_normals(normal_depth_image, frame.info,
                                              mask)

    features = to_color_feature(
        frame.rgb_image, blur=blur, color_space=color_space)

    return frame, features




def load_sample1_dataset():
    return load_ftb(Path(__file__).parent / "../test-data/rgbd/sample1")


def load_sample2_dataset():
    return load_ftb(Path(__file__).parent / "../test-data/rgbd/sample1")
