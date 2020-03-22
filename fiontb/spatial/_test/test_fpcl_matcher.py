import unittest
from pathlib import Path

import torch

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.spatial.matching import FramePointCloudMatcher, FPCLMatcherOp
from fiontb.testing import prepare_frame, ColorMode


class TestFPCLMatcher(unittest.TestCase):
    def test_operator(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../../test-data/rgbd/sample2")
        frame, color_features = prepare_frame(
            dataset[0], color_mode=ColorMode.RGB)
        FramePointCloud.from_frame(frame)

        fpcl = FramePointCloud.from_frame(frame).to(torch.double)

        target = FramePointCloudMatcher.from_frame_pcl(
            fpcl,
            color_features.to(torch.double)).target
        match = FPCLMatcherOp.Match()

        torch.manual_seed(10)
        source_points = target.points[100:110, 100:110, :].reshape(-1, 3)
        source_points += torch.rand(source_points.size(0),
                                    3, dtype=torch.double).abs() * 0.01
        source_points = source_points.to(torch.double)
        source_points.requires_grad = True

        source_normals = target.normals[100:110, 100:110, :].reshape(-1, 3)

        inputs = (source_points, source_normals, target, match)
        torch.autograd.gradcheck(FPCLMatcherOp.apply, inputs, eps=1e-6, atol=1e-4,
                                 raise_exception=True)
