from pathlib import Path
import math

import open3d

import numpy as np
import cv2
import torch

from fiontb.frame import FramePointCloud

from .result import ICPResult


class RGBDOdometry:
    def __init__(self, color_only=False, iterations=None):
        self.color_only = color_only

        if iterations is None:
            iterations = [20, 10, 5]
        self.option = open3d.odometry.OdometryOption()
        self.option.iteration_number_per_pyramid_level = open3d.IntVector(
            iterations)

    def estimate_frame(self, source_frame, target_frame, transform=None, **kwargs):
        if self.color_only:
            jacobian_term = open3d.odometry.RGBDOdometryJacobianFromColorTerm()
        else:
            jacobian_term = open3d.odometry.RGBDOdometryJacobianFromHybridTerm()

        rgbd_img_s = open3d.geometry.RGBDImage()

        rgbd_img_s.depth = open3d.geometry.Image(
            source_frame.depth_image.astype(np.float32) * source_frame.info.depth_scale)
        rgbd_img_s.color = open3d.geometry.Image(
            cv2.cvtColor(source_frame.rgb_image,
                         cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0)

        rgbd_img_t = open3d.geometry.RGBDImage()
        rgbd_img_t.depth = open3d.geometry.Image(
            target_frame.depth_image.astype(np.float32) * target_frame.info.depth_scale)
        rgbd_img_t.color = open3d.geometry.Image(
            cv2.cvtColor(target_frame.rgb_image,
                         cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0)

        kcam = target_frame.info.kcam
        intrinsic = open3d.camera.PinholeCameraIntrinsic(
            target_frame.depth_image.shape[1], target_frame.depth_image.shape[0],
            kcam.matrix[0, 0], kcam.matrix[1, 1],
            kcam.matrix[0, 2], kcam.matrix[1, 2])

        if transform is not None:
            transform = transform.cpu().numpy()
        else:
            transform = np.identity(4)

        is_good, transform, hessian = open3d.odometry.compute_rgbd_odometry(
            rgbd_img_s, rgbd_img_t, intrinsic, jacobian=jacobian_term,
            option=self.option)

        transform = torch.from_numpy(transform)
        hessian = torch.from_numpy(hessian)

        return ICPResult(transform, hessian,
                         0.0 if is_good else 1.0,
                         0.0 if is_good else 1.0,
                         1.0 if is_good else 0.0,
                         1.0 if is_good else 0.0)


class ColorICP:
    def __init__(self, radius_iters):
        self.radius_iters = radius_iters

    def estimate_frame(self, source_frame, target_frame, transform=None, **kwargs):
        source_pcl = FramePointCloud.from_frame(source_frame).unordered_point_cloud(
            world_space=False).to_open3d()
        target_pcl = FramePointCloud.from_frame(target_frame).unordered_point_cloud(
            world_space=False).to_open3d()

        if transform is None:
            transform = np.identity(4)
        else:
            transform = transform.cpu().numpy()

        for radius, iters in self.radius_iters:
            if radius < math.inf:
                source_down = open3d.geometry.voxel_down_sample(
                    source_pcl, radius)
                target_down = open3d.geometry.voxel_down_sample(
                    target_pcl, radius)
                print(source_down)
            else:
                source_down = source_pcl
                target_down = target_pcl

            open3d.estimate_normals(
                source_down,
                search_param=open3d.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            open3d.estimate_normals(
                target_down,
                search_param=open3d.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            result = open3d.registration.registration_colored_icp(
                source_down, target_down, radius, transform,
                open3d.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=iters))
            transform = result.transformation

        transform = torch.from_numpy(result.transformation)
        return ICPResult(transform, torch.eye(4),
                         1.0 - result.inlier_rmse,
                         1.0 - result.inlier_rmse,
                         1.0, 1.0)


class Tests:
    _TEST_DATA = Path(__file__).parent / "../../test-data/rgbd"

    @staticmethod
    def _test(odometry, dataset):
        from fiontb.viz.show import show_pcls
        from fiontb.testing import prepare_frame

        frame0, _ = prepare_frame(dataset[0], filter_depth=True)
        frame1, _ = prepare_frame(dataset[1], filter_depth=True)
        result = odometry.estimate_frame(frame1, frame0)

        pcl_params = {
            'world_space': False, 'compute_normals': False
        }

        frame1 = FramePointCloud.from_frame(
            frame1).unordered_point_cloud(**pcl_params)
        show_pcls(
            [FramePointCloud.from_frame(frame0).unordered_point_cloud(**pcl_params),
             frame1, frame1.transform(result.transform.float())])

    def rgbd_real(self):
        from fiontb.data.ftb import load_ftb
        dataset = load_ftb(Tests._TEST_DATA / "sample1")
        Tests._test(RGBDOdometry(False), dataset)

    def rgbd_synthetic(self):
        from fiontb.data.ftb import load_ftb
        dataset = load_ftb(Tests._TEST_DATA / "sample2")
        Tests._test(RGBDOdometry(False), dataset)

    def rgbd_trajectory(self):
        from fiontb.viz.show import show_pcls
        from fiontb.testing import prepare_frame
        from fiontb.data.ftb import load_ftb
        from fiontb.camera import RTCamera

        dataset = load_ftb(Tests._TEST_DATA / "sample1")

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
        from fiontb.data.ftb import load_ftb
        dataset = load_ftb(Tests._TEST_DATA / "sample1")
        Tests._test(ColorICP([(math.inf, 50)]), dataset)

    def coloricp_synthetic(self):
        from fiontb.data.ftb import load_ftb
        dataset = load_ftb(Tests._TEST_DATA / "sample2")
        Tests._test(ColorICP([
            (0.04, 50), (0.02, 30), (0.01, 14)
        ]), dataset)


def _main():
    import fire

    fire.Fire(Tests)


if __name__ == '__main__':
    _main()
