"""Test the ICP algorithm variants.
"""
from pathlib import Path

import torch
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
_SAMPLE = "sample1"


class Tests:
    def geometric1(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = ICPOdometry(25)

        frame, _ = prepare_frame(dataset[0], filter_depth=True)
        next_frame, _ = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt, tracking_ok = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_mask=fpcl.mask,
            target_normals=fpcl.normals)

        print(tracking_ok)
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True, blur=True)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=True)

        import matplotlib.pyplot as plt
        # plt.imshow(features0.cpu().transpose(1, 0).transpose(1, 2))
        # plt.show()
        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = ICPOdometry(100)
        relative_rt, tracking_ok = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                                source_feats=features1.to(
                                                    device),
                                                target_points=fpcl.points,
                                                target_normals=fpcl.normals,
                                                target_mask=fpcl.mask,
                                                target_feats=features0.to(
                                                    device),
                                                geom_weight=0.0,
                                                feat_weight=1.0)
        print(tracking_ok)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def hybrid(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True, blur=True)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = ICPOdometry(100)
        relative_rt, tracking_ok = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                                source_feats=features1.to(
                                                    device),
                                                target_points=fpcl.points,
                                                target_normals=fpcl.normals,
                                                target_mask=fpcl.mask,
                                                target_feats=features0.to(
                                                    device),
                                                geom_weight=0.5,
                                                feat_weight=0.5)

        print(tracking_ok)
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_geometric(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = MultiscaleICPOdometry([
            (0.25, 20, False),
            (0.5, 20, False),
            (1.0, 20, False)
        ], lost_track_threshold=1e-1)

        frame, _ = prepare_frame(dataset[0], filter_depth=True)
        next_frame, _ = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt, tracking_ok = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_normals=fpcl.normals,
            target_mask=fpcl.mask)

        print(tracking_ok)
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_hybrid(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True, blur=False)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=False)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = MultiscaleICPOdometry([
            (0.25, 5, True),
            (0.5, 25, True),
            (0.75, 25, True),
            (1.0, 50, True)
        ])
        relative_rt, tracking_ok = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                                source_feats=features1.to(
                                                    device),
                                                target_points=fpcl.points,
                                                target_normals=fpcl.normals,
                                                target_mask=fpcl.mask,
                                                target_feats=features0.to(
                                                    device),
                                                geom_weight=0.5,
                                                feat_weight=0.5)
        print(tracking_ok)
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def fail(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = ICPOdometry(10)

        last_frame_num = len(dataset) - 1
        frame, _ = prepare_frame(dataset[0], filter_depth=False)
        next_frame, _ = prepare_frame(
            dataset[last_frame_num], filter_depth=False)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt, tracking_ok = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_mask=fpcl.mask,
            target_normals=fpcl.normals)

        print(tracking_ok)
        evaluate(dataset, relative_rt, last_frame_num)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(Tests)
