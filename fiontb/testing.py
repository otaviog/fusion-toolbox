import cv2
import numpy as np

from fiontb.filtering import bilateral_depth_filter


def prepare_frame(frame, scale=1, filter_depth=True):
    height, width = frame.depth_image.shape
    height, width = int(height*scale), int(width*scale)

    if filter_depth:
        frame.depth_image = bilateral_depth_filter(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale).numpy()

    frame.depth_image = cv2.resize(frame.depth_image.astype(
        np.uint16), (width, height)).astype(np.int32)
    frame.rgb_image = cv2.resize(cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2HSV),
                                 (width, height))
    frame.info.kcam = frame.info.kcam.scaled(scale)

    return frame
