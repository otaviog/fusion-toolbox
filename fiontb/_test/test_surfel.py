from pathlib import Path

import fire
import tenviz
import torch
import quaternion
import numpy as np

from fiontb.frame import FramePointCloud
from fiontb.viz.surfelrender import show_surfels
from fiontb.data.ftb import load_ftb
from ..surfel import SurfelModel, SurfelCloud


class TestDatatype:
    def adding(self):
        device = "cpu:0"
        test_data = Path(__file__).parent / "../../test-data/rgbd"
        dataset = load_ftb(test_data / "sample2")

        live_surfels = SurfelCloud.from_frame_pcl(
            FramePointCloud.from_frame(dataset[0]))

        gl_context = tenviz.Context()
        model = SurfelModel(gl_context, live_surfels.size)
        model.add_surfels(live_surfels, update_gl=True)
        show_surfels(gl_context, [model])

    def downsample(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../test-data/rgbd/sample2")

        live_surfels = SurfelCloud.from_frame(dataset[0])

        downsampled = live_surfels.downsample(0.05)

        show_surfels(tenviz.Context(), [live_surfels, downsampled])

    def transform(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../test-data/rgbd/sample2")
        surfels0 = SurfelCloud.from_frame(dataset[0])
        surfels1 = surfels0.clone()

        transformation = torch.eye(4)
        np.random.seed(5)
        transformation[:3, :3] = torch.from_numpy(quaternion.as_rotation_matrix(
            quaternion.from_euler_angles(np.random.rand(3))))
        transformation[0, 3] = 2
        transformation[1, 3] = 0
        transformation[2, 3] = 2

        surfels1.itransform(transformation)

        show_surfels(tenviz.Context(), [surfels0, surfels1, surfels0.transform(transformation)])


if __name__ == '__main__':
    fire.Fire(TestDatatype)
