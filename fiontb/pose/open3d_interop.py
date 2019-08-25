from pathlib import Path

import open3d

import numpy as np
import cv2
import torch


def estimate_odometry(source_frame, target_frame, color_only=False):
    if color_only:
        jacobian_term = open3d.odometry.RGBDOdometryJacobianFromColorTerm()
    else:
        jacobian_term = open3d.odometry.RGBDOdometryJacobianFromHybridTerm()

    rgbd_img_s = open3d.geometry.RGBDImage()

    rgbd_img_s.depth = open3d.geometry.Image(
        source_frame.depth_image.astype(np.float32) * source_frame.info.depth_scale)
    rgbd_img_s.color = open3d.geometry.Image(
        cv2.cvtColor(source_frame.rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0)

    rgbd_img_t = open3d.geometry.RGBDImage()
    rgbd_img_t.depth = open3d.geometry.Image(
        target_frame.depth_image.astype(np.float32) * target_frame.info.depth_scale)
    rgbd_img_t.color = open3d.geometry.Image(
        cv2.cvtColor(target_frame.rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0)

    kcam = target_frame.info.kcam
    intrinsic = open3d.camera.PinholeCameraIntrinsic(
        target_frame.depth_image.shape[1], target_frame.depth_image.shape[0],
        kcam.matrix[0, 0], kcam.matrix[1, 1],
        kcam.matrix[0, 2], kcam.matrix[1, 2])

    is_good, transform, hessian = open3d.odometry.compute_rgbd_odometry(
        rgbd_img_t, rgbd_img_s, intrinsic, jacobian=jacobian_term)

    return torch.from_numpy(transform).float()


_TEST_DATA = Path(__file__).parent / "_test"


def _test_simple(dataset, color_only):
    from fiontb.viz.show import show_pcls
    from fiontb.frame import FramePointCloud

    transform = estimate_odometry(dataset[0], dataset[1], color_only)
    abs_pose = transform @ dataset[0].info.rt_cam.cam_to_world

    print(abs_pose - dataset[1].info.rt_cam.cam_to_world)

    pcl_opts = dict(world_space=False, compute_normals=False)

    def to_pcl(frame):
        return FramePointCloud.from_frame(
            frame).unordered_point_cloud(**pcl_opts)

    show_pcls(
        [to_pcl(dataset[0]),
         to_pcl(dataset[1]),
         to_pcl(dataset[1]).transform(transform)])


def _test_real():
    from fiontb.data.ftb import load_ftb
    dataset = load_ftb(_TEST_DATA / "sample1")
    _test_simple(dataset, False)


def _test_syn():
    from fiontb.data.ftb import load_ftb
    dataset = load_ftb(_TEST_DATA / "sample2")
    _test_simple(dataset, True)


def _test_dataset():
    from fiontb.data.ftb import load_ftb
    from fiontb.viz.datasetviewer import DatasetViewer

    dataset = load_ftb(_TEST_DATA / "sample1")

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    for i in range(1, len(dataset)):
        frame = dataset[i]
        prev_frame = dataset[i-1]
        relative_t = estimate_odometry(prev_frame, frame)
        dataset.get_info(
            i).rt_cam = prev_frame.info.rt_cam.integrate(relative_t)

    viewer = DatasetViewer(dataset)
    viewer.run()


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    tests = {
        'real': _test_real,
        'syn': _test_syn,
        'dataset': _test_dataset
    }

    parser.add_argument("test", choices=list(tests.keys()))
    args = parser.parse_args()

    tests[args.test]()


if __name__ == '__main__':
    _main()
