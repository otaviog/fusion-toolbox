"""Interactive testing of autograd ICP.
"""

from pathlib import Path

import fire

from fiontb.data.ftb import load_ftb
from fiontb.registration.autogradicp import (AutogradICP, MultiscaleAutogradICP,
                                             AGICPOption)

from fiontb.testing import ColorSpace, preprocess_frame
from .testing import run_trajectory_test, run_pair_test

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame1_idx=5, color_space=ColorSpace.LAB,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=29,
                       color_space=ColorSpace.LAB,
                       blur=True, filter_depth=True)


class _Tests:
    """Tests the AutogradICP class.
    """

    def depth_real(self):
        """Use only depth information of a real scene.
        """
        import math
        run_pair_test(
            AutogradICP(100, geom_weight=1, feat_weight=0,
                        distance_threshold=10, normals_angle_thresh=math.pi/2),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def depth_synthetic(self):
        """Use only depth information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgb_real(self):
        """Use only RGB information of a real scene.
        """
        run_pair_test(
            AutogradICP(500, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgb_synthetic(self):
        """Use only RGB information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgbd_real(self):
        """Use RGB+depth information of a real scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgbd_synthetic(self):
        """Use RGB+depth information of a synthetic scene.
        """
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_depth_real(self):
        """Use multiscale depth information of a real scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_depth_synthetic(self):
        """Use multiscale depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgb_real(self):
        """Use multiscale RGB information of a real scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 5, 0.05, geom_weight=0, feat_weight=1)
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgb_synthetic(self):
        """Use multiscale RGB information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 5, 0.05, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgbd_real(self):
        """Use multiscale RGB+depth information of a real scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 15, 0.05, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=10, feat_weight=1),
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgbd_synthetic(self):
        """Use multiscale RGB+depth information of a synthetic scene.
        """
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 10, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, geom_weight=10, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def trajectory(self):
        """Test mulstiscale RGB and depth alignment on a a synthetic trajectory.
        """

        icp = MultiscaleAutogradICP([
            AGICPOption(1.0, 50, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0)
        ])

        dataset = load_ftb(_TEST_DATA / "sample1")
        run_trajectory_test(icp, dataset, color_space=ColorSpace.RGB,
                            blur=False)

    def pcl_rgbd_real(self):
        """Test using sparse point cloud.
        """
        from fiontb.pointcloud import PointCloud
        from fiontb.spatial.matching import PointCloudMatcher
        from fiontb.viz import geoshow
        from fiontb.camera import RTCamera
        from fiontb._utils import profile

        dataset = load_ftb(_TEST_DATA / "sample1")

        frame0, features0 = preprocess_frame(dataset[0], color_space=ColorSpace.LAB,
                                             blur=False, filter_depth=True)
        frame1, features1 = preprocess_frame(dataset[29], color_space=ColorSpace.LAB,
                                             blur=False, filter_depth=True)

        icp = AutogradICP(30, learning_rate=1,
                          geom_weight=0.5, feat_weight=0.1)

        device = "cuda:0"

        pcl0, mask = PointCloud.from_frame(frame0, world_space=False)
        pcl0 = pcl0.to(device)
        features0 = features0[:, mask].view(-1, pcl0.size).to(device)

        pcl1, mask = PointCloud.from_frame(frame1, world_space=False)
        pcl1 = pcl1.to(device)
        features1 = features1[:, mask].view(-1, pcl1.size).to(device)

        matcher = PointCloudMatcher.from_point_cloud(
            pcl0, features0,
            distance_upper_bound=0.5
        )
        with profile(Path(__file__).parent / "pcl_rgbd_real.prof"):
            result = icp.estimate2(pcl1.points, pcl1.normals, features1,
                                   matcher)

        gt_traj = {0: frame0.info.rt_cam, 1: frame1.info.rt_cam}
        pred_traj = {0: RTCamera(), 1: RTCamera(result.transform)}

        pcl2 = pcl1.transform(relative_rt.to(device))
        geoshow([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(_Tests)
