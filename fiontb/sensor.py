"""Sensor IO.
"""

from enum import Enum
import numpy as np

import onireader

from fiontb.camera import KCamera
from fiontb.frame import Frame, FrameInfo


class DeviceType(Enum):
    """Sensor device kind selector.
    """
    ASUS_XTION = 1


DEVICE_TO_KCAM = {
    DeviceType.ASUS_XTION: np.array([[544.47329, 0.0, 320],
                                     [0.0, 544.47329, 240],
                                     [0.0, 0.0, 1.0]])
}


class Sensor:
    """Sensor IO based on OpenNI2.
    """

    def __init__(self, device: onireader.Device, device_type=None):
        self.device = device
        if device_type is None:
            device_type = DeviceType.ASUS_XTION
        kmatrix = DEVICE_TO_KCAM.get(device_type, None)
        self.kcam = KCamera(kmatrix)

    def next_frame(self):
        """Reads the next frame from the device stream.

        Returns: (:obj:`fiontb.frame.Frame`): Next frame.

        """
        # pylint: disable=unused-variable

        depth_img, depth_ts, depth_idx = self.device.readDepth()
        rgb_img, rgb_ts, rgb_idx = self.device.readColor()

        info = FrameInfo(self.kcam, depth_scale=1, depth_bias=0.0, depth_max=16000,
                         timestamp=depth_ts)

        frame = Frame(info, depth_img, rgb_image=rgb_img)

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
