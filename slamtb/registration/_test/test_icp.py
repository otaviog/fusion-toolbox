"""Interactive testing of ICPOdometry.
"""
from pathlib import Path

import fire

from slamtb.data import set_start_at_eye
from slamtb.data.ftb import load_ftb
from slamtb.processing import ColorSpace
from slamtb.registration import MultiscaleRegistration
from slamtb.registration.icp import ICPOdometry

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
            ICPOdometry(15, geom_weight=1, feat_weight=0,
                        distance_threshold=10, normals_angle_thresh=2),
            set_start_at_eye(load_ftb(_TEST_DATA / "sample1")),
            **REAL_FRAME_ARGS,
            device="cpu")

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
            MultiscaleRegistration([
                (1.0, ICPOdometry(15, geom_weight=1, feat_weight=0)),
                (0.5, ICPOdometry(10, geom_weight=1, feat_weight=0)),
                (0.5, ICPOdometry(5, geom_weight=1, feat_weight=0))]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_depth_synthetic():
        """Use multiscale depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, ICPOdometry(15, geom_weight=1, feat_weight=0)),
                (0.5, ICPOdometry(20, geom_weight=1, feat_weight=0)),
                (0.5, ICPOdometry(20, geom_weight=1, feat_weight=0))]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgb_real():
        """Use multiscale RGB information of a real scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, ICPOdometry(20, geom_weight=0, feat_weight=1)),
                (0.5, ICPOdometry(10, geom_weight=0, feat_weight=1)),
                (0.5, ICPOdometry(5, geom_weight=0, feat_weight=1))]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_rgb_synthetic():
        """Use multiscale RGB information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, ICPOdometry(20, geom_weight=0, feat_weight=1)),
                (0.5, ICPOdometry(10, geom_weight=0, feat_weight=1)),
                (0.5, ICPOdometry(5, geom_weight=0, feat_weight=1))]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgbd_real():
        """Use multiscale RGB+depth information of a real scene.
        """
        print("HERE")
        run_pair_test(
            MultiscaleRegistration([
                (1.0, ICPOdometry(20, geom_weight=1, feat_weight=1)),
                (0.5, ICPOdometry(20, geom_weight=1, feat_weight=1)),
                (0.5, ICPOdometry(30, geom_weight=1, feat_weight=1))
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
            MultiscaleRegistration([
                (1.0, ICPOdometry(10, geom_weight=geom_weight,
                                  feat_weight=feat_weight)),
                (0.5, ICPOdometry(20, geom_weight=geom_weight,
                                  feat_weight=feat_weight)),
                (0.5, ICPOdometry(20, geom_weight=geom_weight,
                                  feat_weight=feat_weight)),
                (0.5, ICPOdometry(30, geom_weight=geom_weight,
                                  feat_weight=feat_weight))]),
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
        run_pair_test(
            ICPOdometry(20, geom_weight=10, feat_weight=1, so3=True),
            load_ftb(_TEST_DATA / "sample2"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def trajectory():
        """Test mulstiscale RGB and depth alignment on a a synthetic trajectory.
        """

        dataset = load_ftb(_TEST_DATA / "sample2")
        icp = MultiscaleRegistration([
            (1.0, ICPOdometry(10, geom_weight=10, feat_weight=1)),
            (0.5, ICPOdometry(10, geom_weight=10, feat_weight=1)),
            (0.5, ICPOdometry(10, geom_weight=10, feat_weight=1))])
        icp = ICPOdometry(25, geom_weight=1, feat_weight=0)
        run_trajectory_test(icp, dataset,
                            filter_depth=SYNTHETIC_FRAME_ARGS['filter_depth'],
                            blur=SYNTHETIC_FRAME_ARGS['blur'],
                            color_space=SYNTHETIC_FRAME_ARGS['color_space'])


if __name__ == '__main__':
    fire.Fire(_Tests)
