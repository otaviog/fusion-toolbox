"""Interactive testing of ICPOdometry.
"""
from pathlib import Path

import torch
import fire

from fiontb.data import set_start_at_eye
from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.registration.icp import (
    ICPOdometry, MultiscaleICPOdometry, ICPVerifier, ICPOption)
from fiontb.testing import preprocess_frame, ColorSpace
from fiontb.viz.show import geoshow

from .testing import run_trajectory_test, run_pair_test

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame0_idx=0, frame1_idx=14, color_space=ColorSpace.GRAY,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=14, color_space=ColorSpace.LAB,
                       blur=True, filter_depth=True)


class _Tests:
    """Tests ICPOdometry class.
    """

    @staticmethod
    def depth_real():
        """Use only depth information of a real scene.
        """
        run_pair_test(
            ICPOdometry(15, geom_weight=1, feat_weight=0),
            set_start_at_eye(load_ftb(_TEST_DATA / "sample1")),
            **REAL_FRAME_ARGS,
            device="cuda:0")

    @staticmethod
    def depth_synthetic():
        """Use only depth information of a synthetic scene.
        """
        run_pair_test(
            ICPOdometry(15, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def rgb_real():
        """Use only RGB information of a real scene.
        """
        run_pair_test(
            ICPOdometry(300, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def rgb_synthetic():
        """Use only RGB information of a synthetic scene.
        """
        run_pair_test(
            ICPOdometry(300, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def rgbd_real():
        """Use RGB+depth information of a real scene.
        """
        run_pair_test(
            ICPOdometry(40, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def rgbd_synthetic():
        """Use RGB+depth information of a synthetic scene.
        """
        run_pair_test(
            ICPOdometry(40, geom_weight=1, feat_weight=1),
            load_ftb(
                _TEST_DATA / "sample2"
            ),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_depth_real():
        """Use multiscale depth information of a real scene.
        """
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_depth_synthetic():
        """Use multiscale depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 20, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 20, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 20, geom_weight=1, feat_weight=0),
                ICPOption(0.5, 20, geom_weight=1, feat_weight=0)]),
            #load_ftb(_TEST_DATA / "sample2"),
            load_ftb(
                "/home/otaviog/3drec/slam-feature/data/replica/replica-ftb/hotel_0"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgb_real():
        """Use multiscale RGB information of a real scene.
        """
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 5, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_rgb_synthetic():
        """Use multiscale RGB information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=0, feat_weight=1),
                ICPOption(0.5, 5, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgbd_real():
        """Use multiscale RGB+depth information of a real scene.
        """
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 20, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 15, geom_weight=10, feat_weight=1),
                ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
                # ICPOption(1, 10, geom_weight=0, feat_weight=1, so3=True)
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_rgbd_synthetic():
        """Use multiscale RGB+depth information of a synthetic scene.
        """
        geom_weight = 10
        feat_weight = 0
        run_pair_test(
            MultiscaleICPOdometry([
                ICPOption(1.0, 10, geom_weight=geom_weight,
                          feat_weight=feat_weight),
                ICPOption(0.5, 20, geom_weight=geom_weight,
                          feat_weight=feat_weight),
                ICPOption(0.5, 20, geom_weight=geom_weight,
                          feat_weight=feat_weight),
                ICPOption(0.5, 30, geom_weight=geom_weight,
                          feat_weight=feat_weight)
            ]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def fail():
        """Test for fail alignment.
        """
        run_pair_test(ICPOdometry(10), load_ftb(_TEST_DATA / "sample1"))

    @staticmethod
    def so3():
        """Test rotation only alignment.
        """
        dataset = load_ftb(_TEST_DATA / "sample1")
        dataset = set_start_at_eye(dataset)
        device = "cuda:0"

        icp_verifier = ICPVerifier()
        frame, features0 = prepare_frame(
            dataset[0], scale=1, filter_depth=True, color_space=ColorSpace.RGB, blur=False)
        next_frame, features1 = prepare_frame(
            dataset[6], scale=1, filter_depth=True, color_space=ColorSpace.RGB, blur=False)

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

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.to(device))
        pcl1 = pcl1.transform(init_transform)

        geoshow([pcl0, pcl1.transform(init_transform), pcl2])

    @staticmethod
    def trajectory():
        """Test mulstiscale RGB and depth alignment on a a synthetic trajectory.
        """

        dataset = load_ftb(_TEST_DATA / "sample2")
        icp = MultiscaleICPOdometry([
            ICPOption(1.0, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            #ICPOption(1.0, 10, feat_weight=1, so3=True),
        ])
        icp = ICPOdometry(25, geom_weight=1, feat_weight=0)
        run_trajectory_test(icp, dataset,
                            filter_depth=SYNTHETIC_FRAME_ARGS['filter_depth'],
                            blur=SYNTHETIC_FRAME_ARGS['blur'],
                            color_space=SYNTHETIC_FRAME_ARGS['color_space'])


if __name__ == '__main__':
    fire.Fire(_Tests)
