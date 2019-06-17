"""Common filtering of 3d reconstruction for frames.
"""

import numpy as np
import torch

from fiontb._cfiontb import bilateral_filter_depth_image as _bilateral_filter_depth_image


def bilateral_filter_depth_image(depth, mask, filter_width=6,
                                 sigma_d=4.50000000225,
                                 sigma_r=29.9999880000072):
    return _bilateral_filter_depth_image(torch.from_numpy(depth),
                                         torch.from_numpy(mask).byte(),
                                         filter_width, sigma_d, sigma_r).numpy()


def _test():
    from pathlib import Path
    import argparse

    import cv2
    import matplotlib.pyplot as plt

    depth = cv2.imread(str(Path(__file__).parent / "_tests/assets" / "frame_depth.png"),
                       cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(str(Path(__file__).parent / "_tests/assets" / "frame_mask.png"),
                      cv2.IMREAD_ANYDEPTH)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("test", choices=["bilateral"])

    args = arg_parser.parse_args()

    plt.figure()
    plt.title("input")
    plt.imshow(depth)
    if args.test == "bilateral":
        filter_depth = bilateral_filter_depth_image(
            depth.astype(np.int32), depth > 0,
            13, 4.50000000225,
            29.9999880000072)

    filtered_depth_image = cv2.bilateralFilter(
        depth.astype(np.float32),
        13, 4.50000000225,
        29.9999880000072)
    plt.figure()
    plt.title("cv2")
    plt.imshow(filtered_depth_image)

    plt.figure()
    plt.title(args.test)
    plt.imshow(filter_depth)

    plt.show()


if __name__ == '__main__':
    _test()
