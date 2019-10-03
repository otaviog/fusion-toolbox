"""Test the ICP algorithm variants.
"""
from pathlib import Path

import torch
from torchvision.transforms import ToTensor
import fire

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls
from fiontb.pose.icp import (ICPOdometry, MultiscaleICPOdometry)
from fiontb.testing import prepare_frame

from ._utils import evaluate

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

torch.set_printoptions(precision=10)

_OTHER_FRAME_INDEX = 5
_SAMPLE = "sample2"

_TO_TENSOR = ToTensor()


class Tests:
    def geometric1(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = ICPOdometry(25)

        frame = prepare_frame(dataset[0], filter_depth=True)
        next_frame = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_mask=fpcl.mask,
            target_normals=fpcl.normals)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame = prepare_frame(dataset[0], filter_depth=True, to_hsv=True)
        next_frame = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        image = _TO_TENSOR(frame.rgb_image).to(device)
        next_image = _TO_TENSOR(next_frame.rgb_image).to(device)

        icp = ICPOdometry(100)
        relative_rt = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                   target_points=fpcl.points,
                                   target_normals=fpcl.normals,
                                   target_mask=fpcl.mask,
                                   target_feats=image,
                                   source_feats=next_image.flatten(),
                                   geom_weight=0.001,
                                   feat_weight=1.0)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_geometric(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = MultiscaleICPOdometry([
            (0.25, 25, False),
            (0.5, 25, False),
            (0.75, 25, False),
            (1.0, 25, False)
        ])

        frame = prepare_frame(dataset[0], filter_depth=True)
        next_frame = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_normals=fpcl.normals,
            target_mask=fpcl.mask)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_color(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame = prepare_frame(dataset[0], filter_depth=True, to_hsv=True)
        next_frame = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        image = _TO_TENSOR(frame.rgb_image).to(device)
        next_image = _TO_TENSOR(next_frame.rgb_image).to(device)

        icp = MultiscaleICPOdometry([
            (0.25, 25, True),
            (0.5, 25, True),
            (0.75, 25, True),
            (1.0, 25, True)
        ])
        relative_rt = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                   target_points=fpcl.points,
                                   target_normals=fpcl.normals,
                                   target_mask=fpcl.mask,
                                   target_feats=image,
                                   source_feats=next_image,
                                   geom_weight=0.5,
                                   feat_weight=0.5)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(Tests)
