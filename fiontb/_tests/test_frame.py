"""Tests fiontb.frame module.
"""

import unittest
from pathlib import Path

import numpy as np
import cv2
import torch

from fiontb.camera import KCamera
from fiontb.frame import FrameInfo, estimate_normals, EstimateNormalsMethod
from fiontb.ui import convert_normals_to_rgb
import tenviz

# pylint: disable=missing-docstring, no-self-use


class TestFrame(unittest.TestCase):

    def test_compute_normals(self):
        depth_image = torch.from_numpy(
            cv2.imread(str(Path(__file__).parent / "assets" / "frame_depth.png"),
                       cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth_image > 0
        info = FrameInfo(KCamera(np.array([[544.47327, 0., 320.],
                                           [0., 544.47327, 240.],
                                           [0., 0., 1.]])), 0.001, 0.0)

        normals_cpu = estimate_normals(depth_image, info, mask)
        normals_gpu = estimate_normals(
            depth_image.to("cuda:0"), info, mask.to("cuda:0"))

        torch.testing.assert_allclose(normals_gpu.cpu(), normals_cpu,
                                      1.0, 0.0)
