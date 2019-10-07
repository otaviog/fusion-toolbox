from pathlib import Path
from cProfile import Profile

import torch
from torchvision.transforms.functional import to_tensor
import fire

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls
from fiontb.testing import prepare_frame
from fiontb.pose.autogradicp import AutogradICP, MultiscaleAutogradICP
from fiontb._utils import profile as _profile
from ._utils import evaluate

# pylint: disable=no-self-use

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

torch.set_printoptions(precision=10)

_OTHER_FRAME_INDEX = 5
_SAMPLE = "sample1"


class _Tests:
    def geometric(self, profile=False):
        device = "cuda:0"
        icp = AutogradICP(25, 0.05)

        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        frame = prepare_frame(dataset[0], filter_depth=True)
        next_frame = prepare_frame(dataset[_OTHER_FRAME_INDEX],
                                   filter_depth=True)

        fpcl0 = FramePointCloud.from_frame(frame).to(device)
        fpcl1 = FramePointCloud.from_frame(next_frame).to(device)

        with _profile(Path(__file__).parent /
                      "autogradicp-geometric.prof",
                      really=profile):
            relative_rt = icp.estimate(fpcl1.kcam,
                                       source_points=fpcl1.points,
                                       source_mask=fpcl1.mask,
                                       target_points=fpcl0.points,
                                       target_mask=fpcl0.mask,
                                       target_normals=fpcl0.normals)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl0.unordered_point_cloud(world_space=False)
        pcl1 = fpcl1.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt)

        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        device = "cuda:0"
        icp = AutogradICP(100, 0.05)

        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True, blur=True)
        next_frame, features1 = prepare_frame(dataset[_OTHER_FRAME_INDEX],
                                              filter_depth=True, to_hsv=True, blur=True)

        fpcl0 = FramePointCloud.from_frame(frame).to(device)
        fpcl1 = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt = icp.estimate(fpcl1.kcam.to(device),
                                   source_points=fpcl1.points,
                                   source_mask=fpcl1.mask,
                                   target_points=fpcl0.points,
                                   target_mask=fpcl0.mask,
                                   target_feats=features0.to(device),
                                   source_feats=features1.to(device),
                                   geom_weight=0, feat_weight=1)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl0.unordered_point_cloud(world_space=False)
        pcl1 = fpcl1.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt)
        show_pcls([pcl0, pcl1, pcl2])

    def hybrid(self):
        device = "cuda:0"
        icp = AutogradICP(50, 0.05)

        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True, blur=True)
        next_frame, features1 = prepare_frame(dataset[_OTHER_FRAME_INDEX],
                                              filter_depth=True, to_hsv=True, blur=True)

        fpcl0 = FramePointCloud.from_frame(frame).to(device)
        fpcl1 = FramePointCloud.from_frame(next_frame).to(device)
        pcl1 = fpcl1.unordered_point_cloud(world_space=False)

        image0 = to_tensor(frame.rgb_image).to(device)
        image1 = to_tensor(next_frame.rgb_image).to(device)

        relative_rt = icp.estimate(
            fpcl1.kcam, source_points=fpcl1.points,
            source_mask=fpcl1.mask,
            target_points=fpcl0.points, target_mask=fpcl0.mask,
            target_normals=fpcl0.normals,
            source_feats=image1, target_feats=image0,
            geom_weight=0.5, feat_weight=0.5)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl0.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt)

        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_geometric(self):
        device = "cuda:0"
        icp = MultiscaleAutogradICP([(0.5, 100, 0.05, False),
                                     (1.0, 100, 0.05, False)
                                     ])

        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        frame, _ = prepare_frame(dataset[0], filter_depth=True)
        next_frame, _ = prepare_frame(dataset[_OTHER_FRAME_INDEX],
                                      filter_depth=True)

        fpcl0 = FramePointCloud.from_frame(frame).to(device)
        fpcl1 = FramePointCloud.from_frame(next_frame).to(device)

        relative_rt = icp.estimate(
            fpcl1.kcam, source_points=fpcl1.points,
            source_mask=fpcl1.mask,
            target_points=fpcl0.points, target_mask=fpcl0.mask,
            target_normals=fpcl0.normals,
            geom_weight=1.0, feat_weight=0.0)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl0.unordered_point_cloud(world_space=False)
        pcl1 = fpcl1.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt)

        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_hybrid(self):
        device = "cuda:0"
        icp = MultiscaleAutogradICP([(0.5, 50, 0.05, True),
                                     (1.0, 50, 0.05, True)])

        dataset = load_ftb(_TEST_DATA / _SAMPLE)

        frame, features0 = prepare_frame(
            dataset[0], filter_depth=True, to_hsv=True)
        next_frame, features1 = prepare_frame(dataset[_OTHER_FRAME_INDEX],
                                              filter_depth=True, to_hsv=True)

        fpcl0 = FramePointCloud.from_frame(frame).to(device)
        fpcl1 = FramePointCloud.from_frame(next_frame).to(device)
        pcl1 = fpcl1.unordered_point_cloud(world_space=False)

        relative_rt = icp.estimate(
            fpcl1.kcam, source_points=fpcl1.points,
            source_mask=fpcl1.mask,
            target_points=fpcl0.points, target_mask=fpcl0.mask,
            target_normals=fpcl0.normals,
            source_feats=features1.to(device),
            target_feats=features0.to(device),
            geom_weight=0.5, feat_weight=0.5)

        evaluate(dataset, relative_rt, _OTHER_FRAME_INDEX)

        pcl0 = fpcl0.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt)

        show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(_Tests)
