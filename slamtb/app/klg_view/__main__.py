"""Visualizer for KLG files"""

import argparse

import cv2

from slamtb.data.klg import KLG
from slamtb.viz.datasetviewer import DatasetViewer
from slamtb.sensor import PresetIntrinsics, get_sensor_kcamera
from slamtb.camera import KCamera

_SENSOR_TYPE_MAP = {'xtion': (PresetIntrinsics.ASUS_XTION, 0.001)}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_klg", metavar="input-klg",
                        help="Input KLG file")
    parser.add_argument("--sensor-type", '-t', choices=list(_SENSOR_TYPE_MAP.keys()),
                        help="Recording sensor type")
    parser.add_argument("--depth-scale", '-s', type=float,
                        help="Override sensor depth scale")

    args = parser.parse_args()

    dataset = KLG(args.input_klg)

    if args.sensor_type is not None:
        dev, scale = _SENSOR_TYPE_MAP[args.sensor_type]
        dataset.depth_scale = scale
        frame0 = dataset[0]
        height, width = frame0.depth_image.shape[:2]
        dataset.kcam = get_sensor_kcamera(dev, width, height)

    if args.depth_scale is not None:
        dataset.depth_scale = args.depth_scale

    ds_viewer = DatasetViewer(dataset, args.input_klg)

    ds_viewer.run()


if __name__ == '__main__':
    _main()
