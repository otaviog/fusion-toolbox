"""General testing facilitators for intern fusion toolbox tests.

"""

from enum import Enum

import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor

from fiontb.frame import estimate_normals
from fiontb.filtering import bilateral_depth_filter


class ColorMode(Enum):
    RGB = 0,
    GRAY = 1
    HSV = 2


def prepare_frame(frame, scale=1, filter_depth=True, color_mode=ColorMode.HSV,
                  compute_normals=False):
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
            mask,
            depth_scale=frame.info.depth_scale).numpy()

    if compute_normals:
        normal_depth_image = frame.depth_image
        if not filter_depth:
            normal_depth_image = bilateral_depth_filter(
                frame.depth_image.to("cuda:0"),
                mask.to("cuda:0"), depth_scale=frame.info.depth_scale).numpy()

        frame.normal_image = estimate_normals(normal_depth_image, frame.info,
                                              mask)

    features = get_color_feature(
        frame.rgb_image, color_mode)

    return frame, to_tensor(features)


def get_color_feature(rgb_image, blur=False, color_mode=ColorMode.HSV):
    features = rgb_image
    if blur:
        features = cv2.blur(features, (5, 5))

    if color_mode == ColorMode.HSV:
        features = cv2.cvtColor(features, cv2.COLOR_RGB2HSV)
    elif color_mode == ColorMode.GRAY:
        features = cv2.cvtColor(features, cv2.COLOR_RGB2GRAY)

    return to_tensor(features)
