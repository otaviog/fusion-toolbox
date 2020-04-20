"""Interactive testing of autograd ICP.
"""

from pathlib import Path
import math

import numpy as np
import fire

from slamtb.data.ftb import load_ftb
from slamtb.registration import MultiscaleRegistration
from slamtb.registration.autogradicp import AutogradICP
from slamtb.testing import ColorSpace

from .testing import (run_trajectory_test,
                      run_pair_test,
                      run_pcl_pair_test)

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame0_idx=0, frame1_idx=3, color_space=ColorSpace.GRAY,
                            blur=False, filter_depth=False, device="cuda:0",
                            view_matrix=np.array(
                                [[-0.997461, 0, -0.0712193, 0.612169],
                                 [-0.0168819, 0.971499, 0.23644, -1.29119],
                                 [0.0691895, 0.237042, -0.969033, -0.336442],
                                 [0, 0, 0, 1]]))

REAL_FRAME_ARGS = dict(frame1_idx=8,
                       color_space=ColorSpace.LAB,
                       blur=True, filter_depth=True)


class _Tests:
    """Tests the AutogradICP class.
    """

    @staticmethod
    def depth_real():
        """Use only depth information of a real scene.
        """
        run_pair_test(
            AutogradICP(600, geom_weight=1, feat_weight=0,
                        learning_rate=0.01,
                        distance_threshold=10,
                        normals_angle_thresh=math.pi),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def depth_synthetic():
        """Use only depth information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(500, learning_rate=0.1, geom_weight=1, feat_weight=0,
                        distance_threshold=0.1,
                        normals_angle_thresh=math.pi/4.0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def rgb_real():
        """Use only RGB information of a real scene.
        """
        run_pair_test(
            AutogradICP(600, learning_rate=0.1,
                        geom_weight=0, feat_weight=1.0,
                        distance_threshold=0.1, normals_angle_thresh=math.pi/4),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def rgb_synthetic():
        """Use only RGB information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=0, feat_weight=1,
                        huber_loss_alpha=2,
                        distance_threshold=0.1, normals_angle_thresh=math.pi/4,
                        feat_residual_thresh=-0.005),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def rgbd_real():
        """Use RGB+depth information of a real scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def rgbd_synthetic():
        """Use RGB+depth information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_depth_real():
        """Use multiscale depth information of a real scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, AutogradICP(15, geom_weight=1, feat_weight=0)),
                (0.5, AutogradICP(10, geom_weight=1, feat_weight=0)),
                (0.5, AutogradICP(5, geom_weight=1, feat_weight=0))]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_depth_synthetic():
        """Use multiscale depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, AutogradICP(15, geom_weight=1, feat_weight=0)),
                (0.5, AutogradICP(10, geom_weight=1, feat_weight=0)),
                (0.5, AutogradICP(5, geom_weight=1, feat_weight=0))]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgb_real():
        """Use multiscale RGB information of a real scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, AutogradICP(20, 0.05, geom_weight=0, feat_weight=1)),
                (0.5, AutogradICP(10, 0.05, geom_weight=0, feat_weight=1)),
                (0.5, AutogradICP(5, 0.05, geom_weight=0, feat_weight=1))
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_rgb_synthetic():
        """Use multiscale RGB information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, AutogradICP(20, 0.05, geom_weight=0, feat_weight=1)),
                (0.5, AutogradICP(10, 0.05, geom_weight=0, feat_weight=1)),
                (0.5, AutogradICP(5, 0.05, geom_weight=0, feat_weight=1))]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def ms_rgbd_real():
        """Use multiscale RGB+depth information of a real scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                (1.0, AutogradICP(20, 0.05, geom_weight=10, feat_weight=1)),
                (0.5, AutogradICP(15, 0.05, geom_weight=10, feat_weight=1)),
                (0.5, AutogradICP(10, 0.05, geom_weight=10, feat_weight=1))]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    @staticmethod
    def ms_rgbd_synthetic():
        """Use multiscale RGB+depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleRegistration([
                AutogradICP(1.0, 10, geom_weight=10, feat_weight=1),
                AutogradICP(0.5, 10, geom_weight=10, feat_weight=1),
                AutogradICP(0.5, 10, geom_weight=10, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    @staticmethod
    def trajectory():
        """Test mulstiscale RGB and depth alignment on a a synthetic trajectory.
        """

        icp = MultiscaleRegistration([
            (1.0, AutogradICP(50, 0.05, geom_weight=1, feat_weight=0)),
            (0.5, AutogradICP(50, 0.05, geom_weight=1, feat_weight=0)),
            (0.5, AutogradICP(50, 0.05, geom_weight=1, feat_weight=0))
        ])

        dataset = load_ftb(_TEST_DATA / "sample1")
        run_trajectory_test(icp, dataset, color_space=ColorSpace.RGB,
                            blur=False)

    @staticmethod
    def pcl_rgb_real():
        """Test using sparse point cloud.
        """

        dataset = load_ftb(_TEST_DATA / "sample1")
        icp = AutogradICP(
            600, learning_rate=0.1,
            geom_weight=0, feat_weight=1.0, huber_loss_alpha=2,
            distance_threshold=0.1, normals_angle_thresh=math.pi/4,
            feat_residual_thresh=0.005)

        run_pcl_pair_test(icp, dataset,
                          profile_file=Path(__file__).parent /
                          "pcl_rgbd_real.prof",
                          filter_depth=True,
                          blur=True,
                          color_space=ColorSpace.LAB,
                          frame0_idx=0,
                          frame1_idx=8,
                          device="cuda:0")

    @staticmethod
    def pcl_ms_rgbd_real():
        """Test using sparse point cloud.
        """

        dataset = load_ftb(_TEST_DATA / "sample1")

        icp = MultiscaleRegistration(
            [(-1, AutogradICP(600, learning_rate=0.1,
                              geom_weight=10, feat_weight=1.0, huber_loss_alpha=4,
                              distance_threshold=0.1, normals_angle_thresh=math.pi/4,
                              feat_residual_thresh=0.005)),
             (0.025, AutogradICP(300, learning_rate=0.1,
                                 geom_weight=10, feat_weight=1.0, huber_loss_alpha=4,
                                 distance_threshold=0.1, normals_angle_thresh=math.pi/4,
                                 feat_residual_thresh=0.5)),
             (0.05, AutogradICP(300, learning_rate=0.1,
                                geom_weight=10, feat_weight=1.0, huber_loss_alpha=4,
                                distance_threshold=0.1, normals_angle_thresh=math.pi/4,
                                feat_residual_thresh=0.5))])

        run_pcl_pair_test(icp, dataset,
                          profile_file=Path(__file__).parent /
                          "pcl_rgbd_real.prof",
                          filter_depth=True,
                          blur=False,
                          color_space=ColorSpace.LAB,
                          frame0_idx=0,
                          frame1_idx=8,
                          device="cuda:0")


if __name__ == '__main__':
    fire.Fire(_Tests)
