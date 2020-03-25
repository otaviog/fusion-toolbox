from pathlib import Path

import fire
import torch
import quaternion
import numpy as np

import tenviz

from slamtb.frame import FramePointCloud
from slamtb.viz.surfelrender import show_surfels
from slamtb.data.ftb import load_ftb
from ..surfel import SurfelModel, SurfelCloud, SurfelVolume


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

        show_surfels(tenviz.Context(), [
                     surfels0, surfels1, surfels0.transform(transformation)])

    def downsample(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../test-data/rgbd/sample2")

        live_surfels = SurfelCloud.from_frame(dataset[0])
        live_surfels.features = torch.rand(16, live_surfels.size)

        downsampled = live_surfels.downsample(0.05)
        print(live_surfels.features[:, 100:105])
        print(downsampled.features[:, 100:105])
        show_surfels(tenviz.Context(), [live_surfels, downsampled])

    def merge(self):
        from slamtb.data import set_start_at_eye
        dataset = set_start_at_eye(load_ftb(Path(__file__).parent /
                                            "../../test-data/rgbd/sample2"))
        frame0 = dataset[0]
        frame1 = dataset[14]

        surfels0 = SurfelCloud.from_frame(frame0)
        surfels0.features = torch.rand(16, surfels0.size)
        surfels0.itransform(frame0.info.rt_cam.cam_to_world)

        surfels1 = SurfelCloud.from_frame(frame1)
        surfels1.itransform(frame1.info.rt_cam.cam_to_world)
        surfels1.features = torch.rand(16, surfels1.size)

        volume = SurfelVolume(torch.tensor([[-10, -10, -10],
                                            [10, 10, 10]]),
                              0.001, 16)
        volume.merge(surfels0)
        volume.merge(surfels1)

        surfels2 = volume.to_surfel_cloud()

        print(surfels0.features[:, 100:105])
        print(surfels1.features[:, 100:105])
        print(surfels2.features[:, 100:105])
        
        show_surfels(tenviz.Context(),
                     [surfels0, surfels1, surfels2])


if __name__ == '__main__':
    fire.Fire(TestDatatype)
