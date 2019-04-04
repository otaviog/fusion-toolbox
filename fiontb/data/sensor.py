from enum import Enum
import numpy as np
import onireader

from fiontb.camera import KCamera

from .datatype import Snapshot


class DeviceType(Enum):
    ASUS_XTION = 1

DEVICE_TO_KCAM = {
    DeviceType.ASUS_XTION: np.array([[544.47329, 0.0, 320],
                          [0.0, 544.47329, 240],
                          [0.0, 0.0, 1.0]])
}

class Sensor:
    def __init__(self, device_uri, device_type):
        self.device = onireader.Device(device)
        kcam = DEVICE_TO_KCAM.get(device_type, None)
        self.kcam = KCamera(kcam)
        
    def next_frame(self):
        depth_img, depth_ts, depth_idx = self.device.readDepth()
        rgb_img, rgb_ts, rgb_idx = self.device.readColor()

        snap = Snapshot(depth_img, self.kcam, depth_scale, depth_bias=0.0, depth_max=16000,
                        rgb_image=rgb_img, timestamp=depth_ts)

        return snap


class DatasetSensor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_idx = 0

    def next_frame(self):
        if self.current_idx >= len(self.dataset):
            return None, False

        snap = self.dataset[self.current_idx]
        self.current_idx += 1

        return snap, True
