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

        self.option.iteration_number_per_pyramid_level = open3d.utility.IntVector(
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
                         1.0 if is_good else 0.0)


class ColorICP:
    def __init__(self, scale_iters):
        self.scales = [scale for scale, _ in scale_iters]
        self.iters = [iters
                      for _, iters in scale_iters]

    def estimate_frame(self, source_frame, target_frame, transform=None, **kwargs):
        if transform is None:
            transform = np.identity(4)
        else:
            transform = transform.cpu().numpy()

        source_pyramid = FramePointCloud.from_frame(
            source_frame).pyramid(self.scales)
        target_pyramid = FramePointCloud.from_frame(
            target_frame).pyramid(self.scales)

        max_corresp_dist = 1
        for iters, source_pyr, target_pyr in zip(
                self.iters, source_pyramid, target_pyramid):

            source_pcl = source_pyr.unordered_point_cloud(
                world_space=False).to_open3d()
            target_pcl = target_pyr.unordered_point_cloud(
                world_space=False).to_open3d()

            conv_criteria = open3d.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=iters)
            result = open3d.registration.registration_colored_icp(
                source_pcl, target_pcl, max_corresp_dist, transform,
                conv_criteria)
            transform = result.transformation

        transform = torch.from_numpy(result.transformation)
        return ICPResult(transform, torch.eye(4),
                         result.inlier_rmse,
                         1.0)

    def estimate_pcl(self, source_pcl, target_pcl, transform=None, **kwargs):
        source_pcl = source_pcl.to_open3d()
        target_pcl = target_pcl.to_open3d()

        if transform is None:
            transform = np.eye(4)
        for radius, iters in zip(self.scales, self.iters):
            source_down = source_pcl.voxel_down_sample(radius)
            target_down = target_pcl.voxel_down_sample(radius)

            if False:
                open3d.estimate_normals(
                    source_down,
                    open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                open3d.estimate_normals(
                    target_down,
                    open3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            conv_criteria = open3d.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=iters)
            result = open3d.registration.registration_colored_icp(
                source_pcl, target_pcl, radius, transform,
                conv_criteria)

            transform = result.transformation

        transform = torch.from_numpy(result.transformation)
        return ICPResult(transform, torch.eye(6, dtype=transform.dtype),
                         result.inlier_rmse,
                         1.0)
