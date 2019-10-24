"""Test the ICP algorithm variants.
"""
from pathlib import Path

import torch
import fire

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls
from fiontb.pose.icp import (
    ICPOdometry, MultiscaleICPOdometry, ICPVerifier, ICPOption)
from fiontb.testing import prepare_frame
from fiontb._utils import profile
from fiontb.camera import RTCamera

from ._utils import evaluate

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"
_FRAME_INDEX = 0
_OTHER_FRAME_INDEX = 4
_SAMPLE = "sample1"
_BLUR = True

ICP_VERIFIER = ICPVerifier()

torch.set_printoptions(precision=10)


class Tests:
    def geometric(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = ICPOdometry(25)

        prev_frame, _ = prepare_frame(dataset[_FRAME_INDEX], filter_depth=True)
        next_frame, _ = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        prev_fpcl = FramePointCloud.from_frame(prev_frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        result = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=prev_fpcl.points,
            target_mask=prev_fpcl.mask,
            target_normals=prev_fpcl.normals,
            feat_weight=0, geom_weight=1)
        print(result.transform)
        relative_rt = result.transform
        print("Tracking: ", ICP_VERIFIER(result))
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = prev_fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = ICPOdometry(100)
        result = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                              source_feats=features1.to(
                                  device),
                              target_points=fpcl.points,
                              target_normals=fpcl.normals,
                              target_mask=fpcl.mask,
                              target_feats=features0.to(
                                  device))
        print("Tracking: ", ICP_VERIFIER(result))

        relative_rt = result.transform
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def hybrid(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = ICPOdometry(60, geom_weight=.5, feat_weight=.5)
        result = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                              source_feats=features1.to(
                                  device),
                              target_points=fpcl.points,
                              target_normals=fpcl.normals,
                              target_mask=fpcl.mask,
                              target_feats=features0.to(
                                  device))

        relative_rt = result.transform
        print("Tracking: ", ICP_VERIFIER(result))

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_geometric(self):
        device = "cuda:0"
        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        icp = MultiscaleICPOdometry([
            ICPOption(1.0, 30, geom_weight=1.0, use_feats=False),
            ICPOption(0.5, 30, geom_weight=1.0, use_feats=False),
            ICPOption(0.5, 30, geom_weight=1.0, use_feats=False)
        ])

        frame, _ = prepare_frame(dataset[_FRAME_INDEX], filter_depth=True)
        next_frame, _ = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        with profile(Path(__file__).parent / "icp-multiscale-geometric.prof"):
            for _ in range(5):
                result = icp.estimate(
                    next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                    target_points=fpcl.points,
                    target_normals=fpcl.normals,
                    target_mask=fpcl.mask)

        relative_rt = result.transform

        print("Tracking: ", ICP_VERIFIER(result))
        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_hybrid(self):
        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        device = "cuda:0"

        frame, features0 = prepare_frame(
            dataset[_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)
        next_frame, features1 = prepare_frame(
            dataset[_OTHER_FRAME_INDEX], filter_depth=True, to_hsv=True, blur=_BLUR)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        icp = MultiscaleICPOdometry([
            ICPOption(1.0, 30, geom_weight=.5, feat_weight=.5, use_feats=True),
            ICPOption(0.5, 30, geom_weight=.5, feat_weight=.5, use_feats=True),
            ICPOption(0.5, 30, geom_weight=.5, feat_weight=.5, use_feats=True),
            ICPOption(1.0, 30, feat_weight=1, use_feats=True, so3=True),
        ])

        with profile(Path(__file__).parent / "icp-multiscale-hybrid.prof"):
            for _ in range(5):
                result = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                                      source_feats=features1.to(
                                          device),
                                      target_points=fpcl.points,
                                      target_normals=fpcl.normals,
                                      target_mask=fpcl.mask,
                                      target_feats=features0.to(
                                          device))

        relative_rt = result.transform
        print("Tracking: ", ICP_VERIFIER(result))

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

        result = icp.estimate(
            next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
            target_points=fpcl.points,
            target_mask=fpcl.mask,
            target_normals=fpcl.normals)

        relative_rt = result.transform
        print("Tracking: ", ICP_VERIFIER(result))

        evaluate(dataset, relative_rt, last_frame_num)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))

        show_pcls([pcl0, pcl1, pcl2])

    def trajectory(self):
        dataset = load_ftb(Path(__file__).parent /
                           "../../../test-data/rgbd/sample1")
        device = "cuda:0"

        icp = MultiscaleICPOdometry([
            ICPOption(1.0, 30, geom_weight=0.5,
                      feat_weight=0.5, use_feats=True),
            ICPOption(0.5, 30, geom_weight=0.5,
                      feat_weight=0.5, use_feats=True),
            ICPOption(0.5, 30, geom_weight=0.5,
                      feat_weight=0.5, use_feats=True),
        ])

        frame_args = {
            'filter_depth': True,
            'blur': True,
            'to_hsv': True
        }
        prev_frame, prev_features = prepare_frame(
            dataset[0], **frame_args)

        pcls = [FramePointCloud.from_frame(
            prev_frame).unordered_point_cloud(world_space=False)]

        accum_pose = RTCamera(dtype=torch.double)

        for i in range(1, len(dataset)):
            next_frame, next_features = prepare_frame(
                dataset[i], **frame_args)

            result = icp.estimate_frame(next_frame, prev_frame,
                                        source_feats=next_features.to(device),
                                        target_feats=prev_features.to(device),
                                        device=device)
            print("{} tracking {}".format(i, ICP_VERIFIER(result)))
            accum_pose = accum_pose.integrate(result.transform.cpu().double())

            pcl = FramePointCloud.from_frame(
                next_frame).unordered_point_cloud(world_space=False)
            pcl = pcl.transform(accum_pose.matrix.float())
            pcls.append(pcl)

            prev_frame, prev_features = next_frame, next_features
        show_pcls(pcls)


if __name__ == '__main__':
    fire.Fire(Tests)
