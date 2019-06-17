"""Tests fiontb.frame module.
"""

import unittest
from pathlib import Path

import numpy as np
import cv2
import torch

from fiontb.camera import KCamera
from fiontb.frame import FrameInfo, compute_normals
from fiontb.ui import convert_normals_to_rgb
import tenviz

# pylint: disable=missing-docstring, no-self-use


class TestFrame(unittest.TestCase):

    def test_compute_normals(self):

        depth_image = cv2.imread(str(Path(__file__).parent / "assets" / "frame_depth.png"),
                                 cv2.IMREAD_ANYDEPTH)

        kcam = KCamera(np.array([[544.47327, 0., 320.],
                                 [0., 544.47327, 240.],
                                 [0., 0., 1.]]))

        info = FrameInfo(kcam, 0.001, 0.0)

        normals = compute_normals(depth_image, info, depth_image > 0)
        normals = convert_normals_to_rgb(normals)
        normals = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(__file__).parent /
                        "out-frame-normals.png"), normals)
