"""Tests fiontb.frame module.
"""

import unittest
from pathlib import Path

import fire
import numpy as np
import cv2
import torch
import tenviz

from fiontb.camera import KCamera
from fiontb.frame import FrameInfo, estimate_normals, EstimateNormalsMethod, FramePointCloud
from fiontb.ui import convert_normals_to_rgb
from fiontb.data.ftb import load_ftb


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


class Tests:
    def normals(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../test-data/rgbd/sample1")

        pcl = FramePointCloud.from_frame(dataset[0]).unordered_point_cloud(
            world_space=False, compute_normals=True)

        context = tenviz.Context()
        with context.current():
            points = tenviz.create_point_cloud(pcl.points, pcl.colors.float()/255, point_size=4)
            
            normals = tenviz.create_quiver(pcl.points, pcl.normals*.005,
                                           torch.ones(pcl.size, 3))
        viewer = context.viewer([points, normals], cam_manip=tenviz.CameraManipulator.WASD)

        viewer.show(1)


if __name__ == '__main__':
    fire.Fire(Tests)
