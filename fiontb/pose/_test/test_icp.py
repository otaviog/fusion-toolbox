"""Test the ICP algorithm variants.
"""
from pathlib import Path

import torch
import fire
from tqdm import tqdm

from fiontb.data import set_start_at_eye
from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls
from fiontb.pose.icp import (
    ICPOdometry, MultiscaleICPOdometry, ICPVerifier, ICPOption)
from fiontb.testing import prepare_frame
from fiontb._utils import profile
from fiontb.camera import RTCamera

from .testing import (evaluate, evaluate_trajectory,
                      run_trajectory_test, run_pair_test)

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame1_idx=10, to_gray=False, to_hsv=True,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=28, to_gray=True, to_hsv=False,
                       blur=True, filter_depth=True)


class Tests:
    def depth_real(self):
        run_pair_test(
            ICPOdometry(15, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def depth_synthetic(self):
        run_pair_test(
            ICPOdometry(15, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgb_real(self):
        run_pair_test(
            ICPOdometry(300, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgb_synthetic(self):
        run_pair_test(
            ICPOdometry(300, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgbd_real(self):
        run_pair_test(
            ICPOdometry(40, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgbd_synthetic(self):
        run_pair_test(
            ICPOdometry(40, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_depth_real(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_depth_synthetic(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgb_real(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 5, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgb_synthetic(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 5, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgbd_real(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 15, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
                # ICPOption(1, 10, geom_weight=0, feat_weight=1, so3=True)
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgbd_synthetic(self):
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 10, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=10, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def fail(self):
        run_pair_test(ICPOdometry(10), load_ftb(_TEST_DATA / "sample1"))

    def so3(self):
        dataset = load_ftb(_TEST_DATA / "sample1")
        dataset = set_start_at_eye(dataset)
        device = "cuda:0"

        icp_verifier = ICPVerifier()
        frame, features0 = prepare_frame(
            dataset[0], scale=1, filter_depth=True, to_hsv=True, blur=True)
        next_frame, features1 = prepare_frame(
            dataset[6], scale=1, filter_depth=True, to_hsv=True, blur=True)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        init_transform = torch.eye(4, dtype=next_fpcl.rt_cam.matrix.dtype,
                                   device=next_fpcl.device)
        init_transform[: 3, 3] = -next_fpcl.rt_cam.center

        icp = ICPOdometry(100, feat_weight=1.0, so3=True)
        result = icp.estimate(next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
                              source_feats=features1.to(
                                  device),
                              target_points=fpcl.points,
                              target_normals=fpcl.normals,
                              target_mask=fpcl.mask,
                              target_feats=features0.to(
                                  device), transform=init_transform)
        relative_rt = result.transform
        print("Tracking: ", icp_verifier(result))

        evaluate(next_fpcl.rt_cam, fpcl.rt_cam,
                 relative_rt)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        pcl1 = pcl1.transform(init_transform)
        show_pcls([pcl0, pcl1.transform(init_transform), pcl2])

    def trajectory(self):
        dataset = load_ftb(
            Path(__file__).parent / "../../../test-data/rgbd/sample2")
        # "/home/otaviog/3drec/slam-feature/data/scenenn/SceneNN-ftb/045")

        icp = MultiscaleICPOdometry([
            ICPOption(1.0, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            # ICPOption(1.0, 10, feat_weight=1, so3=True),
        ])
        icp = ICPOdometry(15, geom_weight=0, feat_weight=1)
        run_trajectory_test(icp, dataset)


if __name__ == '__main__':
    fire.Fire(Tests)
