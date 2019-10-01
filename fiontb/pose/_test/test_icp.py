from pathlib import Path

import cv2
import torch
from torchvision.transforms import ToTensor
import fire

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_depth_filter
from fiontb.viz.show import show_pcls
from fiontb.pose.icp import (ICPOdometry, MultiscaleICPOdometry)

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"

torch.set_printoptions(precision=10)


def _prepare_frame(frame, bi_filter=True):
    if bi_filter:
        frame.depth_image = bilateral_depth_filter(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale)

    return frame


class Tests:
    def geometric1(self):
        dataset = load_ftb(_TEST_DATA / "sample2")

        icp = ICPOdometry(25)
        device = "cuda:0"

        dataset.get_info(0).rt_cam.matrix = torch.eye(4)

        frame = dataset[0]
        next_frame = dataset[1]

        frame = _prepare_frame(frame, False)
        next_frame = _prepare_frame(next_frame, False)

        fpcl = FramePointCloud.from_frame(frame)
        next_fpcl = FramePointCloud.from_frame(next_frame)

        relative_rt = icp.estimate(fpcl.points.to(device),
                                   fpcl.normals.to(device),
                                   fpcl.mask.to(device),
                                   next_fpcl.points.to(device),
                                   next_fpcl.mask.to(device),
                                   next_fpcl.kcam)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.cpu())
        show_pcls([pcl0, pcl1, pcl2])

    def color(self):
        dataset = load_ftb(_TEST_DATA / "sample2")

        device = "cuda:0"

        dataset.get_info(0).rt_cam.matrix = torch.eye(4)

        frame = dataset[0]
        next_frame = dataset[3]

        frame = _prepare_frame(frame, False)
        next_frame = _prepare_frame(next_frame, False)

        fpcl = FramePointCloud.from_frame(frame)
        next_fpcl = FramePointCloud.from_frame(next_frame)

        to_tensor = ToTensor()
        image = to_tensor(cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2HSV))
        next_image = to_tensor(cv2.cvtColor(
            next_frame.rgb_image, cv2.COLOR_RGB2HSV))

        image = image.to(device)
        next_image = next_image.to(device)

        icp = ICPOdometry(100)
        relative_rt = icp.estimate(fpcl.points.to(device),
                                   fpcl.normals.to(device),
                                   fpcl.mask.to(device),
                                   next_fpcl.points.to(device),
                                   next_fpcl.mask.to(device),
                                   next_fpcl.kcam,
                                   target_feats=image,
                                   source_feats=next_image.flatten(),
                                   geom_weight=0.0,
                                   feat_weight=1.0)

        pcl0 = fpcl.unordered_point_cloud(world_space=False)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.cpu())
        show_pcls([pcl0, pcl1, pcl2])

    def multiscale_geometric(self):
        dataset = load_ftb(_TEST_DATA / "sample2")

        icp = MultiscaleICPOdometry([
            (0.25, 25),
            (0.5, 25),
            (0.75, 25),
            (1.0, 25)
        ])
        device = "cuda:0"

        dataset.get_info(0).rt_cam.matrix = torch.eye(4)

        frame = dataset[0]
        next_frame = dataset[5]

        bilateral_filter = False
        frame = _prepare_frame(frame, bilateral_filter)
        next_frame = _prepare_frame(next_frame, bilateral_filter)

        fpcl = FramePointCloud.from_frame(frame)
        next_fpcl = FramePointCloud.from_frame(next_frame)

        relative_rt = icp.estimate(fpcl.points.to(device),
                                   fpcl.normals.to(device),
                                   fpcl.mask.to(device),
                                   next_fpcl.points.to(device),
                                   next_fpcl.mask.to(device),
                                   next_fpcl.kcam)

        pcl0 = fpcl.unordered_point_cloud(world_space=True)
        pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
        pcl2 = pcl1.transform(relative_rt.cpu())
        show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    fire.Fire(Tests)
