import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import tenviz

from slamtb.processing import (bilateral_depth_filter, erode_mask,
                               estimate_normals, EstimateNormalsMethod)
from slamtb.data.ftb import load_ftb
from slamtb.camera import KCamera
from slamtb.frame import FrameInfo, FramePointCloud

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

    def test_erode_mask(self):
        depth = torch.from_numpy(cv2.imread(
            str(Path(__file__).parent / "assets" / "frame_depth.png"),
            cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth > 0

        cpu_out = erode_mask(mask)
        gpu_out = erode_mask(mask.to("cuda:0"))

        torch.testing.assert_allclose(cpu_out,
                                      gpu_out.cpu(),
                                      1.0, 0.0)

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


class InteractiveTests:
    def bilateral(self):
        """Uses bilateral_depth_filter on a sample image and compares with OpenCV.
        """

        depth = torch.from_numpy(cv2.imread(
            str(Path(__file__).parent / "assets" / "frame_depth.png"),
            cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth > 0

        plt.figure()
        plt.title("input")
        plt.imshow(depth)
        filter_depth = bilateral_depth_filter(
            depth, mask, filter_width=13,
            sigma_color=29.9999880000072,
            sigma_space=4.50000000225)

        filtered_depth_image = cv2.bilateralFilter(
            depth.float().numpy(), 13,
            29.9999880000072, 4.50000000225)
        plt.figure()
        plt.title("cv2")
        plt.imshow(filtered_depth_image)

        plt.figure()
        plt.title("bilateral")
        plt.imshow(filter_depth)

        plt.show()

    def erode_mask(self):
        depth = torch.from_numpy(cv2.imread(
            str(Path(__file__).parent / "assets" / "frame_depth.png"),
            cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth > 0

        out_mask = erode_mask(mask)
        plt.figure()
        plt.title("original")
        plt.imshow(mask)

        plt.figure()
        plt.title("output")
        plt.imshow(out_mask)

        plt.show()

    def normals(self):
        dataset = load_ftb(
            Path(__file__).parent /
            "../../test-data/rgbd/sample1")

        frame0 = dataset[0]
        frame0.depth_image = bilateral_depth_filter(
            frame0.depth_image, frame0.depth_image > 0,
            filter_width=13,
            # depth_scale=frame0.info.depth_scale
        )

        pcl = FramePointCloud.from_frame(frame0).unordered_point_cloud(
            world_space=False, compute_normals=True)

        context = tenviz.Context()
        transform = np.eye(4)
        transform[1, 1] = -1

        with context.current():
            points = tenviz.nodes.create_point_cloud(
                pcl.points, pcl.colors.float()/255, point_size=8)
            points.transform = transform
            normals = tenviz.nodes.create_quiver(pcl.points, pcl.normals*.005,
                                                 torch.ones(pcl.size, 3))
            normals.transform = transform
        context.show([points, normals],
                     cam_manip=tenviz.CameraManipulator.WASD)


if __name__ == '__main__':
    import fire

    fire.Fire(InteractiveTests)
