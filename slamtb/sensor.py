"""Sensor IO.
"""

from enum import Enum
import copy

import numpy as np
import torch

import onireader

from slamtb.camera import KCamera
from slamtb.frame import Frame, FrameInfo


class PresetIntrinsics(Enum):
    """Sensor device kind selector.
    """
    ASUS_XTION = 1


DEVICE_TO_KCAM = {
    PresetIntrinsics.ASUS_XTION: KCamera(torch.Tensor([[544.47329, 0.0, 320],
                                                       [0.0, 544.47329, 240],
                                                       [0.0, 0.0, 1.0]]),
                                         image_size=(640, 480))
}


def get_sensor_kcamera(sensor_type, res_width, res_height):
    """Get a preset sensor calibration.

    This function will scale the intrinsic to the target frame dimension.

    Args:

        sensor_type (PresetIntrinsics): Sensor selector.

        res_width (int): Target frame width in pixels.

        res_height (int): Target frame height in pixels.

    Returns: (:obj:`slamtb.camera.KCamera`): KCamera with the
     hardcoded sensor information.

    """

    kcam = DEVICE_TO_KCAM[sensor_type]

    return kcam.scaled(res_width / kcam.image_size[0], res_height / kcam.image_size[1])


class Sensor:
    """Sensor IO based on OpenNI2.
    """

    def __init__(self, device, preset_intrinsics=None, depth_cutoff=None):
        self.device = device
        depth_vmode = self.device.get_depth_video_mode()
        depth_scale = 0.001
        if depth_vmode.format == onireader.PixelFormat.DEPTH_1_MM:
            depth_scale = 0.001
        elif depth_vmode.format == onireader.PixelFormat.DEPTH_100_UM:
            depth_scale = 0.01

        if preset_intrinsics is None:
            kcam = KCamera.from_estimation_by_fov(device.get_horizontal_fov(),
                                                  device.get_vertical_fov(),
                                                  depth_vmode.width, depth_vmode.height)

            self.base_info = FrameInfo(kcam=kcam,
                                       depth_scale=depth_scale,
                                       depth_bias=0.0,
                                       depth_max=device.get_max_depth_value())
        else:
            self.base_info = FrameInfo(
                kcam=get_sensor_kcamera(preset_intrinsics, depth_vmode.width,
                                        depth_vmode.height),
                depth_scale=depth_scale,
                depth_bias=0.0,
                depth_max=device.get_max_depth_value())

        self.depth_cutoff = depth_cutoff

    def next_frame(self):
        """Reads the next frame from the device stream.

        Returns: (:obj:`slamtb.frame.Frame`): Next frame.

        """
        # pylint: disable=unused-variable

        depth_img, depth_ts, depth_idx = self.device.read_depth()
        rgb_img, rgb_ts, rgb_idx = self.device.read_color()

        info = copy.copy(self.base_info)
        info.timestamp = depth_ts

        if self.depth_cutoff is not None:
            depth_img[depth_img > self.depth_cutoff] = 0

        frame = Frame(info, depth_img.astype(np.int32), rgb_image=rgb_img)

        return frame


class DatasetSensor:
    """Simulates a sensor using a dataset instance.
    """

    def __init__(self, dataset, start_idx=0, prefetch_size=16, num_workers=4):
        """
        Args:

            dataset (object): Any slamtb dataset see `slamtb.data`.
        """

        self.dataset = dataset
        self.current_idx = start_idx
        # self.loader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=prefetch_size,
        #     shuffle=False, num_workers=num_workers)
        # self.loader_iter = iter(self.loader)
        
    def next_frame(self):
        """Reads the next frame from the dataset.

        Returns: (:obj:`slamtb.frame.Frame`): Next frame.

        """
        if self.current_idx >= len(self.dataset):
            return None

        frame = self.dataset[self.current_idx]
        self.current_idx += 1

        return frame
