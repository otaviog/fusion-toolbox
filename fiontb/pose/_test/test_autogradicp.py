from pathlib import Path
from cProfile import Profile

import torch
from torchvision.transforms import ToTensor
import cv2

from fiontb.data.ftb import load_ftb
from fiontb.filtering import bilateral_depth_filter
from fiontb.frame import FramePointCloud
from fiontb.viz.show import show_pcls

from fiontb.pose.autogradicp import AutogradICP

_TEST_DATA = Path(__file__).parent

torch.set_printoptions(precision=10)


def _prepare_frame(frame, bi_filter=True):

    if bi_filter:
        frame.depth_image = bilateral_depth_filter(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale)

    return frame

# pylint: disable=no-self-use


class _Tests:
    def geometric(self, profile=False):
        device = "cuda:0"
        icp = AutogradICP(100, 0.05)

        dataset = load_ftb(_TEST_DATA / "sample2")

        frame = _prepare_frame(dataset[0], bi_filter=False)
        next_frame = _prepare_frame(dataset[2], bi_filter=False)

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        if profile:
            prof = Profile()
            prof.enable()

        relative_rt = icp.estimate(next_fpcl.kcam,
                                   tgt_image_p3d=fpcl.points,
                                   tgt_mask=fpcl.mask,
                                   tgt_normals=fpcl.normals,
                                   src_points=next_fpcl.points)

        if profile:
            prof.disable()
            prof.dump_stats("cprofile.prof")

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.clone()

        pcl2 = pcl2.transform(relative_rt)
        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        device = "cuda:0"
        icp = AutogradICP(100, 0.05)

        dataset = load_ftb(_TEST_DATA / "sample2")

        to_tensor = ToTensor()

        frame = dataset[0]
        next_frame = dataset[1]

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        image = to_tensor(cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2HSV)).to(device)

        next_image = to_tensor(cv2.cvtColor(
            next_frame.rgb_image, cv2.COLOR_RGB2HSV)).to(device)

        relative_rt = icp.estimate(next_fpcl.kcam.to(device),
                                   src_points=next_fpcl.points,
                                   tgt_image_feat=image,
                                   src_image_feat=next_image)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.clone()

        pcl2 = pcl2.transform(relative_rt)
        show_pcls([pcl0, pcl1, pcl2])

    def hybrid(self):
        device = "cuda:0"
        icp = AutogradICP(100, 0.05)

        dataset = load_ftb(_TEST_DATA / "sample2")

        to_tensor = ToTensor()

        frame = dataset[0]
        next_frame = dataset[1]

        fpcl = FramePointCloud.from_frame(frame).to(device)
        next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

        image = to_tensor(cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2HSV)).to(device)

        next_image = to_tensor(cv2.cvtColor(
            next_frame.rgb_image, cv2.COLOR_RGB2HSV)).to(device)
        relative_rt = icp.estimate(next_fpcl.kcam,
                                   src_points=next_fpcl.points,
                                   tgt_image_p3d=fpcl.points,
                                   tgt_mask=fpcl.mask,
                                   tgt_normals=fpcl.normals,
                                   tgt_image_feat=image,
                                   src_image_feat=next_image, geom_weight=0.8,
                                   feat_weight=0.2)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.clone()

        pcl2 = pcl2.transform(relative_rt)
        show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    import fire
    fire.Fire(_Tests)
