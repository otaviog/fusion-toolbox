from pathlib import Path

import fire

from fiontb.data.ftb import load_ftb
from fiontb.pose.autogradicp import (AutogradICP, MultiscaleAutogradICP,
                                     AGICPOption)

from fiontb.testing import ColorMode, prepare_frame
from .testing import run_trajectory_test, run_pair_test

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame1_idx=5, color_mode=ColorMode.LAB,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=29,
                       color_mode=ColorMode.LAB,
                       blur=True, filter_depth=True)


class _Tests:
    def depth_real(self):
        run_pair_test(
            AutogradICP(50, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def depth_synthetic(self):
        run_pair_test(
            AutogradICP(50, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgb_real(self):
        run_pair_test(
            AutogradICP(50, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgb_synthetic(self):
        run_pair_test(
            AutogradICP(50, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgbd_real(self):
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgbd_synthetic(self):
        run_pair_test(
            AutogradICP(50, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_depth_real(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_depth_synthetic(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 15, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 10, geom_weight=1, feat_weight=0),
                AGICPOption(0.5, 5, geom_weight=1, feat_weight=0)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgb_real(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=0, feat_weight=1),
                # AGICPOption(0.5, 5, 0.05, geom_weight=0, feat_weight=1)
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgb_synthetic(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=0, feat_weight=1),
                AGICPOption(0.5, 5, 0.05, geom_weight=0, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def ms_rgbd_real(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 20, 0.05, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 15, 0.05, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, 0.05, geom_weight=10, feat_weight=1),
            ]),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def ms_rgbd_synthetic(self):
        run_pair_test(
            MultiscaleAutogradICP([
                AGICPOption(1.0, 10, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, geom_weight=10, feat_weight=1),
                AGICPOption(0.5, 10, geom_weight=10, feat_weight=1)]),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def trajectory(self):
        icp = MultiscaleAutogradICP([
            AGICPOption(1.0, 50, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0)
        ])

        dataset = load_ftb(_TEST_DATA / "sample1")
        run_trajectory_test(icp, dataset, to_hsv=False,
                            to_gray=True, blur=True)

    def pcl_rgbd_real(self):
        dataset = load_ftb(_TEST_DATA / "sample1")

        frame0, features0 = prepare_frame(dataset[0], color_mode=ColorMode.LAB,
                                          blur=False, filter_depth=True)
        frame1, features1 = prepare_frame(dataset[29], color_mode=ColorMode.LAB,
                                          blur=False, filter_depth=True)

        icp = AutogradICP(3, learning_rate=1, geom_weight=0.5, feat_weight=0.1)

        from fiontb.pointcloud import PointCloud
        from fiontb.spatial.matching import PointCloudMatcher
        from fiontb.viz import geoshow
        from .testing import evaluate
        from fiontb._utils import profile

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

        relative_rt = result.transform

        evaluate(frame0.info.rt_cam, frame1.info.rt_cam,
                 relative_rt)
        pcl2 = pcl1.transform(relative_rt.to(device))
        geoshow([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(_Tests)
