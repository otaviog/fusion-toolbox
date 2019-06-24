import unittest
from pathlib import Path

import numpy as np
import torch
import cv2

from fiontb.filtering import bilateral_filter_depth_image


class TestFiltering(unittest.TestCase):
    def test_bilateral_filter_gpu(self):
        depth = cv2.imread(str(Path(__file__).parent /
                               "assets" / "frame_depth.png"),
                           cv2.IMREAD_ANYDEPTH)

        depth = torch.from_numpy(depth.astype(np.int32))
        mask = depth > 0

        filter_depth_cpu = bilateral_filter_depth_image(
            depth, mask,
            13, 4.50000000225,
            29.9999880000072)

        filter_depth_gpu = bilateral_filter_depth_image(
            depth.to("cuda:0"), mask.to("cuda:0"),
            13, 4.50000000225,
            29.9999880000072)

        torch.testing.assert_allclose(filter_depth_cpu, filter_depth_gpu.cpu())
