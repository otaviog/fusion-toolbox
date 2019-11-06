from pathlib import Path

import fire

from fiontb.data.ftb import load_ftb
from fiontb.pose.autogradicp import (AutogradICP, MultiscaleAutogradICP,
                                     AGICPOption)

from fiontb.testing import ColorMode
from .testing import run_trajectory_test, run_pair_test

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame1_idx=10, color_mode=ColorMode.GRAY,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=1,
                       color_mode=ColorMode.GRAY,
                       blur=False, filter_depth=True)


class _Tests:
    def depth_real(self):
        run_pair_test(
            AutogradICP(15, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def depth_synthetic(self):
        run_pair_test(
            AutogradICP(15, geom_weight=1, feat_weight=0),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgb_real(self):
        run_pair_test(
            AutogradICP(100, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgb_synthetic(self):
        run_pair_test(
            AutogradICP(300, geom_weight=0, feat_weight=1),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgbd_real(self):
        run_pair_test(
            AutogradICP(40, geom_weight=.5, feat_weight=.5),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgbd_synthetic(self):
        run_pair_test(
            AutogradICP(40, geom_weight=.5, feat_weight=.5),
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


if __name__ == '__main__':
    fire.Fire(_Tests)
