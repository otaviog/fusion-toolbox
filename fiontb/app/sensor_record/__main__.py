"""This application reads a OpenNI2 sensor stream and outputs RGBD to
KLG file.
"""

import argparse

import cv2

from fiontb.sensor import Sensor, DeviceType
from fiontb.data.klg import KLGWriter
from fiontb.ui import FrameUI

_SENSOR_TYPE_MAP = {'xtion': DeviceType.ASUS_XTION}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("sensor_type", metavar="sensor-type",
                        choices=list(_SENSOR_TYPE_MAP.keys()))

    parser.add_argument("device_uri", metavar="device-uri")

    parser.add_argument("output", help="Output KLG file")
    args = parser.parse_args()

    sensor = Sensor(args.device_uri, _SENSOR_TYPE_MAP[args.sensor_type])

    frame_ui = FrameUI("RGBD sensor view")

    with open(args.output, 'wb') as outstream:
        klg_writer = KLGWriter(outstream)

        try:
            while True:
                frame = sensor.next_frame()
                if frame is None:
                    break
                klg_writer.write_frame(frame)
                frame_ui.update(frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break
        except KeyboardInterrupt:
            print("Ctrl-C caught, exiting")
        klg_writer.finish()


if __name__ == '__main__':
    _main()
