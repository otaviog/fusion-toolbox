from pathlib import Path
import argparse

import cv2
import torch
import tenviz
import matplotlib.pyplot as plt

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_filter_depth_image
from fiontb.viz.show import show_pcls

from fiontb.pose.icp import (ICPOdometry, MultiscaleICPOdometry)

_TEST_DATA = Path(__file__).parent

torch.set_printoptions(precision=10)


def _prepare_frame(frame, bi_filter=True):

    if bi_filter:
        frame.depth_image = bilateral_filter_depth_image(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale)

    return frame


def _test_geometric1():
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry(15)
    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    frame = dataset[0]
    next_frame = dataset[1]

    frame = _prepare_frame(frame)
    next_frame = _prepare_frame(next_frame)

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
    pcl2 = pcl1.clone()
    pcl2 = pcl2.transform(relative_rt.cpu())
    show_pcls([pcl0, pcl1, pcl2])


def _test_geometric2():
    from fiontb.viz.datasetviewer import DatasetViewer

    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry(15)
    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)
    prev_frame = _prepare_frame(dataset[0])
    prev_fpcl = FramePointCloud.from_frame(prev_frame)

    for i in range(1, len(dataset)):
        frame = _prepare_frame(dataset[i])
        fpcl = FramePointCloud.from_frame(frame)

        relative_rt = icp.estimate(prev_fpcl.points.to(device),
                                   prev_fpcl.normals.to(device),
                                   prev_fpcl.mask.to(device),
                                   fpcl.points.to(device),
                                   fpcl.mask.to(device),
                                   fpcl.kcam)
        relative_rt = relative_rt.cpu()
        dataset.get_info(
            i).rt_cam = dataset[i-1].info.rt_cam.integrate(relative_rt)

        prev_frame = frame
        prev_fpcl = fpcl

    viewer = DatasetViewer(dataset)
    viewer.run()


def _test_multiscale_geometric():
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = MultiscaleICPOdometry([(0.25, 15), (0.5, 10), (1.0, 5)])
    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    frame = dataset[0]
    next_frame = dataset[1]

    frame = _prepare_frame(frame)
    next_frame = _prepare_frame(next_frame)

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
    pcl1.transform(relative_rt.cpu())
    show_pcls([pcl0, pcl1])


def _test_intensity():
    dataset = load_ftb(_TEST_DATA / "sample2")

    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    frame = dataset[0]
    next_frame = dataset[1]

    frame = _prepare_frame(frame, False)
    next_frame = _prepare_frame(next_frame, False)

    fpcl = FramePointCloud.from_frame(frame)
    next_fpcl = FramePointCloud.from_frame(next_frame)

    image = torch.from_numpy(cv2.cvtColor(
        frame.rgb_image, cv2.COLOR_RGB2GRAY)).float() / 255.0
    next_image = torch.from_numpy(cv2.cvtColor(
        next_frame.rgb_image, cv2.COLOR_RGB2GRAY)).float() / 255.0

    if False:
        plt.figure()
        plt.imshow(next_image)
        plt.figure()
        plt.imshow(image)
        plt.show()

    image = image.to(device)
    next_image = next_image.to(device)

    icp = ICPOdometry(15, use_cpu=True)
    relative_rt = icp.estimate(fpcl.points.to(device),
                               fpcl.normals.to(device),
                               fpcl.mask.to(device),
                               next_fpcl.points.to(device),
                               next_fpcl.mask.to(device),
                               next_fpcl.kcam,
                               target_image=image,
                               source_intensity=next_image.flatten()
                               )

    pcl0 = fpcl.unordered_point_cloud(world_space=False)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl2 = pcl1.clone()
    pcl2.transform(relative_rt.cpu())
    show_pcls([pcl0, pcl1, pcl2])


def _main():
    parser = argparse.ArgumentParser()
    tests = {
        'geometric1': _test_geometric1,
        'geometric2': _test_geometric2,
        'multiscale-geometric': _test_multiscale_geometric,
        'intensity': _test_intensity
    }
    parser.add_argument("test", choices=list(tests.keys()))
    args = parser.parse_args()

    tests[args.test]()


if __name__ == '__main__':
    _main()
