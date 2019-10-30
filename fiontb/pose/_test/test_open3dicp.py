from pathlib import Path
import math

import open3d
import torch

from fiontb.viz.show import show_pcls
from fiontb.testing import prepare_frame
from fiontb.data.ftb import load_ftb
from fiontb.camera import RTCamera
from fiontb.frame import FramePointCloud

from ..open3d_interop import RGBDOdometry, ColorICP
from .testing import run_pair_test

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

SYNTHETIC_FRAME_ARGS = dict(frame1_idx=10, to_gray=False, to_hsv=False,
                            blur=False, filter_depth=False)

REAL_FRAME_ARGS = dict(frame1_idx=28, to_gray=False, to_hsv=False,
                       blur=False, filter_depth=True)


class Tests:

    def rgbd_real(self):
        run_pair_test(RGBDOdometry(), load_ftb(_TEST_DATA / "sample1"),
                      **REAL_FRAME_ARGS)

    def rgbd_synthetic(self):
        run_pair_test(RGBDOdometry(), load_ftb(_TEST_DATA / "sample2"),
                      **SYNTHETIC_FRAME_ARGS)

    def rgb_real(self):
        run_pair_test(
            RGBDOdometry(color_only=True),
            load_ftb(_TEST_DATA / "sample1"),
            **REAL_FRAME_ARGS)

    def rgb_synthetic(self):
        run_pair_test(
            RGBDOdometry(color_only=True),
            load_ftb(_TEST_DATA / "sample2"),
            **SYNTHETIC_FRAME_ARGS)

    def rgbd_trajectory(self):
        dataset = load_ftb(_TEST_DATA / "sample1")

        icp = RGBDOdometry(False)

        frame_args = {
            'filter_depth': True,
            'blur': False,
            'to_hsv': False
        }

        prev_frame, prev_features = prepare_frame(
            dataset[0], **frame_args)

        pcls = [FramePointCloud.from_frame(
            prev_frame).unordered_point_cloud(world_space=False)]

        accum_pose = RTCamera(dtype=torch.double)

        for i in range(1, len(dataset)):
            next_frame, next_features = prepare_frame(
                dataset[i], **frame_args)

            result = icp.estimate_frame(next_frame, prev_frame)

            accum_pose = accum_pose.integrate(result.transform.cpu().double())

            pcl = FramePointCloud.from_frame(
                next_frame).unordered_point_cloud(world_space=False)
            pcl = pcl.transform(accum_pose.matrix.float())
            pcls.append(pcl)

            prev_frame, prev_features = next_frame, next_features
        show_pcls(pcls)

    def coloricp_real(self):
        run_pair_test(ColorICP([(math.inf, 50)]),
                      load_ftb(_TEST_DATA / "sample1"),
                      **REAL_FRAME_ARGS)

    def coloricp_synthetic(self):
        run_pair_test(ColorICP([(0.04, 50), (0.02, 30), (0.01, 14)]),
                      load_ftb(_TEST_DATA / "sample2"),
                      **SYNTHETIC_FRAME_ARGS)


if __name__ == '__main__':
    import fire

    fire.Fire(Tests)
