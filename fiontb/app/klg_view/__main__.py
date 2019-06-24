"""Visualizer for KLG files"""

import argparse

import cv2

from fiontb.data.klg import KLG
from fiontb.viz.datasetviewer import DatasetViewer
from fiontb.sensor import DeviceType, DEVICE_TO_KCAM
from fiontb.camera import KCamera

_SENSOR_TYPE_MAP = {'xtion': (DeviceType.ASUS_XTION, 0.001)}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_klg", metavar="input-klg",
                        help="Input KLG file")
    parser.add_argument("--sensor-type", choices=list(_SENSOR_TYPE_MAP.keys()),
                        help="Recording sensor type")

    args = parser.parse_args()

    dataset = KLG(args.input_klg)

    if args.sensor_type is not None:
        dev, scale = _SENSOR_TYPE_MAP[args.sensor_type]
        dataset.depth_scale = scale
        dataset.kcam = KCamera(DEVICE_TO_KCAM[dev])

    ds_viewer = DatasetViewer(dataset, args.input_klg)

    ds_viewer.run()


if __name__ == '__main__':
    _main()
