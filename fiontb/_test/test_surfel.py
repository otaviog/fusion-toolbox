from pathlib import Path

import fire
import tenviz
import torch

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

        live_surfels = SurfelCloud.from_frame_pcl(
            FramePointCloud.from_frame(dataset[0]))

        downsampled = live_surfels.downsample(0.05)

        gl_context = tenviz.Context()
        show_surfels(gl_context, [live_surfels, downsampled])


if __name__ == '__main__':
    fire.Fire(TestDatatype)
