"""General testing facilitators for intern fusion toolbox tests.

"""

import cv2
import numpy as np

from fiontb.filtering import bilateral_depth_filter


def prepare_frame(frame, scale=1, filter_depth=True, to_hsv=False, blur=False):
    height, width = frame.depth_image.shape
    height, width = int(height*scale), int(width*scale)

    if filter_depth:
        frame.depth_image = bilateral_depth_filter(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale).numpy()

    if scale < 1:
        frame.depth_image = cv2.resize(frame.depth_image.astype(
            np.uint16), (width, height)).astype(np.int32)
        frame.rgb_image = cv2.resize(frame.rgb_image,
                                     (width, height))
        frame.info.kcam = frame.info.kcam.scaled(scale)

    if blur:
        frame.rgb_image = cv2.blur(frame.rgb_image, (5, 5))

    if to_hsv:
        frame.rgb_image = cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2HSV)

    return frame
