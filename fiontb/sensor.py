"""Sensor IO.
"""

from enum import Enum
import copy

import numpy as np
import torch

import onireader

from fiontb.camera import KCamera
from fiontb.frame import Frame, FrameInfo


class PresetIntrinsics(Enum):
    """Sensor device kind selector.
    """
    ASUS_XTION = 1


DEVICE_TO_KCAM = {
    PresetIntrinsics.ASUS_XTION: FrameInfo(
        kcam=KCamera(torch.Tensor([[544.47329, 0.0, 320],
                                   [0.0, 544.47329, 240],
                                   [0.0, 0.0, 1.0]])),
        depth_scale=0.001, depth_bias=0, depth_max=16000)
}


class Sensor:
    """Sensor IO based on OpenNI2.
    """

    def __init__(self, device, preset_intrinsics=None, depth_cutoff=None):
        self.device = device
        if preset_intrinsics is None:
            intrinsics = device.get_intrinsics()

            kcam = KCamera.create_from_params(
                intrinsics.fx, intrinsics.fy,
                (intrinsics.cx, intrinsics.cy))

            depth_scale = 0.001
            depth_format = device.get_depth_video_mode().format
            if depth_format == onireader.PixelFormat.DEPTH_1_MM:
                depth_scale = 0.001
            elif depth_format == onireader.PixelFormat.DEPTH_100_UM:
                depth_scale = 0.01

            self.base_info = FrameInfo(kcam=kcam,
                                       depth_scale=depth_scale,
                                       depth_bias=0.0,
                                       depth_max=device.get_max_depth_value())
        else:
            self.base_info = DEVICE_TO_KCAM.get(preset_intrinsics, None)
        self.depth_cutoff = depth_cutoff

    def next_frame(self):
        """Reads the next frame from the device stream.

        Returns: (:obj:`fiontb.frame.Frame`): Next frame.

        """
        # pylint: disable=unused-variable

        depth_img, depth_ts, depth_idx = self.device.readDepth()
        rgb_img, rgb_ts, rgb_idx = self.device.readColor()

        info = copy.copy(self.base_info)
        info.timestamp = depth_ts

        if self.depth_cutoff is not None:
            depth_img[depth_img > self.depth_cutoff] = 0

        frame = Frame(info, depth_img.astype(np.int32), rgb_image=rgb_img)

        return frame


class DatasetSensor:
    """Simulates a sensor using a dataset instance.
    """

    def __init__(self, dataset, start_idx=0):
        """
        Args:

            dataset (object): Any fiontb dataset see `fiontb.data`.
        """

        self.dataset = dataset
        self.current_idx = start_idx

    def next_frame(self):
        """Reads the next frame from the dataset.

        Returns: (:obj:`fiontb.frame.Frame`): Next frame.

        """
        if self.current_idx >= len(self.dataset):
            return None

        frame = self.dataset[self.current_idx]
        self.current_idx += 1

        return frame
