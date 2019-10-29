from pathlib import Path
from cProfile import Profile

import torch
from torchvision.transforms.functional import to_tensor
import fire

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls
from fiontb.testing import prepare_frame
from fiontb.pose.autogradicp import (AutogradICP, MultiscaleAutogradICP,
                                     AGICPOption)

from ._utils import evaluate, run_trajectory_test, run_pair_test

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

torch.set_printoptions(precision=10)

_OTHER_FRAME_INDEX = 5
_SAMPLE = "sample1"


class _Tests:
    def geometric(self):
        run_pair_test(AutogradICP(25, 0.05, geom_weight=1, feat_weight=0))

    def color(self):
        run_pair_test(AutogradICP(100, 0.05, geom_weight=0, feat_weight=1), frame1_idx=2,
                      to_gray=True, to_hsv=False)

    def hybrid(self):
        run_pair_test(AutogradICP(50, 0.05, geom_weight=.5, feat_weight=.5))

    def multiscale_geometric(self):
        run_pair_test(MultiscaleAutogradICP([
            AGICPOption(1.0, 100, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 100, 0.05, geom_weight=1, feat_weight=0),
            AGICPOption(0.5, 100, 0.05, geom_weight=1, feat_weight=0)]))

    def multiscale_hybrid(self):
        run_pair_test(MultiscaleAutogradICP([
            AGICPOption(1.0, 50, 0.05, geom_weight=10, feat_weight=1),
            AGICPOption(0.5, 50, 0.05, geom_weight=10, feat_weight=1),
            AGICPOption(0.5, 50, 0.05, geom_weight=10, feat_weight=1)
        ]))

    def trajectory(self):
        icp = MultiscaleAutogradICP([
            AGICPOption(1.0, 50, 0.05, geom_weight=1, feat_weight=0),
            # AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0),
            # AGICPOption(0.5, 50, 0.05, geom_weight=1, feat_weight=0)
        ])

        dataset = load_ftb(_TEST_DATA / _SAMPLE)
        run_trajectory_test(icp, dataset, to_hsv=False, to_gray=True)


if __name__ == '__main__':
    fire.Fire(_Tests)
