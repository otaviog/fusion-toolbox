"""Tests fiontb.frame module.
"""

import unittest
from pathlib import Path

import numpy as np
import rflow
import cv2

from fiontb.frame import compute_normals

# pylint: disable=missing-docstring, no-self-use


class TestFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds_g = rflow.open_graph(Path(__file__).parent /
                                "../../test-data/rgbd/scene3", "sample")
        cls.dataset = ds_g.to_ftb.call()

    def test_compute_normals(self):
        frame = self.dataset[0]

        normals = compute_normals(frame.depth_image*frame.info.depth_scale)

        normals = (normals + 1)*0.5*255.0
        normals = cv2.cvtColor(normals.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(__file__).parent / "out-frame-normals.png"), normals)
