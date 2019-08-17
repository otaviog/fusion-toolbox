"""This application reads a OpenNI2 sensor stream and outputs RGBD to
KLG file.
"""
import sys
import argparse

import cv2

import onireader

from fiontb.sensor import Sensor, PresetIntrinsics
from fiontb.data.klg import KLGWriter
from fiontb.ui import FrameUI

_SENSOR_TYPE_MAP = {'xtion': PresetIntrinsics.ASUS_XTION}


def _cap_main(argv):
    parser = argparse.ArgumentParser(description="Capture and output to file")
    parser.add_argument("output", help="Output KLG file")
    parser.add_argument(
        "--device-uri", help="Device URI, if not specified then the first found sensor is used")
    parser.add_argument("--resolution", "-r",
                        help="""Find the best fit mode for the given resolution.
                        Overrides depth-mode and rgb-mode""",
                        type=int, nargs=2)

    parser.add_argument(
        "--depth-mode", '-d', help="Depth video mode index, see the `modes` command",
        type=int, default=-1)
    parser.add_argument(
        "--rgb-mode", '-c', help="RGB video mode index, see the `modes` command",
        type=int, default=-1)
    args = parser.parse_args(argv)

    device = onireader.Device()
    device.open(args.device_uri)
    depth_mode, rgb_mode = args.depth_mode, args.rgb_mode
    if args.resolution:
        depth_mode, rgb_mode = device.find_nearest_vmode(
            args.resolution[0], args.resolution[1])
    device.start(depth_mode, rgb_mode)

    sensor = Sensor(device)
    frame_ui = FrameUI("RGBD sensor view")

    print("Press Ctrl-C (terminal) or ESC (view window) to exit")
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


def _print_modes(sensor_infos):
    for i, sinfo in enumerate(sensor_infos):
        print("{} - {} {} {} {}".format(i, sinfo.width,
                                        sinfo.height, sinfo.fps,
                                        sinfo.format))


def _modes_main(argv):
    parser = argparse.ArgumentParser(description="Capture and output to file")
    parser.add_argument(
        "--device-uri", help="Device URI, if not specified then the first found sensor is used")
    args = parser.parse_args(argv)

    dev = onireader.Device()
    dev.open(args.device_uri)
    print("Depth modes")
    _print_modes(dev.get_depth_video_modes())

    print("RGB modes")
    _print_modes(dev.get_color_video_modes())


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=['cap', 'modes'])
    args = parser.parse_args(sys.argv[1:2])

    argv = sys.argv[2:]
    if args.action == 'cap':
        _cap_main(argv)
    elif args.action == 'modes':
        _modes_main(argv)


if __name__ == '__main__':
    _main()
