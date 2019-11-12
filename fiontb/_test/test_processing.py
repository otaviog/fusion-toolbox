import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from fiontb.filtering import bilateral_depth_filter, FeatureMap

# pylint: disable=no-self-use


class TestFiltering(unittest.TestCase):
    """Tests filtering module
    """

    def test_bilateral_filter_gpu(self):
        """Compares bilateral filter gpu vs cpu.
        """
        depth = cv2.imread(str(Path(__file__).parent /
                               "assets" / "frame_depth.png"),
                           cv2.IMREAD_ANYDEPTH)

        depth = torch.from_numpy(depth.astype(np.int32))
        mask = depth > 0

        filter_depth_cpu = bilateral_depth_filter(
            depth, mask, None,
            13, 4.50000000225,
            29.9999880000072)

        filter_depth_gpu = bilateral_depth_filter(
            depth.to("cuda:0"), mask.to("cuda:0"), None,
            13, 4.50000000225,
            29.9999880000072)

        torch.testing.assert_allclose(filter_depth_cpu,
                                      filter_depth_gpu.cpu(),
                                      1.0, 0.0)

    def test_featuremap(self):
        """Sanity check gradient produced by feature map. The gradient is not
        necessary the same as PyTorch's numerical one.
        """

        feat_map = FeatureMap.apply
        torch.set_printoptions(precision=8)
        torch.manual_seed(10)
        inputs = (torch.rand(16, 24, 32, dtype=torch.double),
                  torch.tensor([[15, 15],
                                [20, 20],
                                [5, 5]], dtype=torch.double, requires_grad=True))
        torch.autograd.gradcheck(feat_map, inputs, eps=1e-6, atol=1e-4,
                                 raise_exception=True)


class InteractiveTests:
    def bilateral(self):
        """Uses bilateral_depth_filter on a sample image and compares with OpenCV.
        """

        depth = torch.from_numpy(cv2.imread(
            str(Path(__file__).parent / "_tests/assets" / "frame_depth.png"),
            cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth > 0

        plt.figure()
        plt.title("input")
        plt.imshow(depth)
        filter_depth = bilateral_depth_filter(
            depth, mask,
            None, 13, 4.50000000225,
            29.9999880000072)

        filtered_depth_image = cv2.bilateralFilter(
            depth.float().numpy(), 13, 4.50000000225,
            29.9999880000072)
        plt.figure()
        plt.title("cv2")
        plt.imshow(filtered_depth_image)

        plt.figure()
        plt.title("bilateral")
        plt.imshow(filter_depth)

        plt.show()


if __name__ == '__main__':
    import fire

    fire.Fire(InteractiveTests)
