"""Pose estimation via iterative closest points algorithm.
"""

import torch

from tenviz.pose import SE3

from fiontb.downsample import (
    downsample_xyz, downsample, DownsampleXYZMethod, DownsampleMethod)
from fiontb._cfiontb import (icp_estimate_jacobian_gpu,
                             icp_estimate_intensity_jacobian_gpu,
                             calc_sobel_gradient_gpu)
from fiontb._utils import empty_ensured_size

# pylint: disable=invalid-name


class ICPOdometry:
    """Point-to-plane iterative closest points
    algorithm.

    Attributes:

        num_iters (int): Number of iterations for the Gauss-Newton
         optimization algorithm.

    """

    def __init__(self, num_iters):
        self.num_iters = num_iters

        self.jacobian = None
        self.residual = None
        self.image_grad = None

    def estimate(self, target_points, target_normals, target_mask,
                 source_points, source_mask, kcam, transform=None,
                 target_image=None, source_intensity=None):
        """Estimate the ICP odometry between a target points and normals in a
        grid against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d **normals** that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A uint8 [WxH] mask
              tensor of valid target points.

            source_points (:obj:`torch.Tensor`): A float [Nx3] tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): A uint8 [N] mask
             tensor of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A float [4x4] initial
             transformation matrix.

        Returns: (:obj:`torch.Tensor`): A [4x4] rigid motion matrix
            that aligns source points to target points.

        """
        assert source_points.is_cuda, "At least source_points must be a cuda tensor."

        device = source_points.device

        kcam = kcam.matrix.to(device)
        if transform is None:
            transform = torch.eye(4, device=device)
        else:
            transform = transform.to(device)

        source_points = source_points.view(-1, 3)
        source_mask = source_mask.view(-1)

        self.jacobian = empty_ensured_size(self.jacobian, source_points.size(0), 6,
                                           device=device, dtype=torch.float)
        self.residual = empty_ensured_size(self.residual, source_points.size(0),
                                           device=device, dtype=torch.float)

        # used only for intensity
        self.image_grad = empty_ensured_size(self.image_grad, target_image.size(0),
                                             target_image.size(1), 2,
                                             device=device, dtype=torch.float)

        geom_only = target_image is None or source_intensity is None

        if not geom_only:
            import matplotlib.pyplot as plt
            calc_sobel_gradient_gpu(target_image, self.image_grad)
            plt.figure()
            plt.imshow(self.image_grad[:, :, 0].cpu())
            plt.figure()
            plt.imshow(self.image_grad[:, :, 1].cpu())
            plt.show()

        for _ in range(self.num_iters):
            if geom_only:
                icp_estimate_jacobian_gpu(target_points, target_normals, target_mask,
                                          source_points, source_mask, kcam, transform,
                                          self.jacobian, self.residual)
            else:
                icp_estimate_intensity_jacobian_gpu(
                    target_points, target_normals, target_image,
                    self.image_grad,
                    target_mask, source_points, source_intensity,
                    source_mask, kcam, transform, self.jacobian,
                    self.residual)

            Jt = self.jacobian.transpose(1, 0)
            JtJ = Jt @ self.jacobian
            upper_JtJ = torch.cholesky(JtJ.double())

            Jr = Jt @ self.residual

            loss = (self.residual*self.residual).mean().item()
            print(loss)

            update = torch.cholesky_solve(
                Jr.view(-1, 1).double(), upper_JtJ).squeeze()

            update_matrix = SE3.exp(
                update.cpu()).to_matrix().to(device).float()
            transform = update_matrix @ transform

        return transform

    def estimate_frame_to_frame(self, target_frame, source_frame, transform=None, use_color=False):

        target_image = None
        source_intensity = None

        return self.estimate(target_frame.points, target_frame.normals,
                             target_frame.mask, source_frame.points, source_frame.mask,
                             source_frame.kcam, transform, target_image, source_intensity)


class IntensityICPOdometry:
    def __init__(self, num_iters, geom_weight=0.5, intensity_weight=0.5):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.intensity_weight = intensity_weight

        self._residual = None
        self._jacobi = None

    def estimate(self, target_points, target_normals, target_intensity, target_mask,
                 source_points, source_intensity, source_mask,
                 kcam, transform=None):
        assert source_points.is_cuda, "At least source_points must be a cuda tensor."

        device = source_points.device

        kcam = kcam.matrix.to(device)
        if transform is None:
            transform = torch.eye(4, device=device)
        else:
            transform = transform.to(device)

        source_points = source_points.view(-1, 3)
        source_mask = source_mask.view(-1)

        self._jacobian = empty_ensured_size(self._jacobian, source_points.size(0), 6,
                                            device=device, dtype=torch.float)
        self._residual = empty_ensured_size(self._residual, source_points.size(0),
                                            device=device, dtype=torch.float)

        for _ in range(self.num_iters):
            icp_estimate_intensity_jacobian_gpu(
                target_points, target_normals, target_mask,
                source_points, source_mask, kcam, transform,
                self._jacobian, self._residual)

            Jt = self._jacobian.transpose(1, 0)
            JtJ = Jt @ self._jacobian
            upper_JtJ = torch.cholesky(JtJ.double())

            Jr = Jt @ self._residual

            update = torch.cholesky_solve(
                Jr.view(-1, 1).double(), upper_JtJ).squeeze()

            update_matrix = SE3.exp(
                update.cpu()).to_matrix().to(device).float()
            transform = update_matrix @ transform

        return transform

    def estimate_frame_to_frame(self, target_frame, source_frame, transform=None):
        return self.estimate(target_frame.points, target_frame.normals,
                             target_frame.mask, source_frame.points, source_frame.mask,
                             source_frame.kcam, transform)


class MultiscaleICPOdometry:
    """Pyramidal point-to-plane iterative closest points
    algorithm.
    """

    def __init__(self, scale_iters, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Initialize the multiscale ICP.

        Args:

            scale_iters (List[(float, int)]): Scale levels and its
             number of iterations. Scales must be <= 1.0

            downsample_xyz_method
             (:obj:`fiontb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the xyz target and source points.
        """

        self.scale_icp = [(scale, ICPOdometry(iters))
                          for scale, iters in scale_iters]
        self.downsample_xyz_method = downsample_xyz_method

    def estimate(self, target_points, target_normals, target_mask,
                 source_points, source_mask,
                 kcam, transform=None):
        """Estimate the ICP odometry between a target frame points and normals
        against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d normals that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A uint8 [WxH] mask tensor
             of valid target points.

            source_points (:obj:`torch.Tensor`): A float [WxHx3] tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): Uint8 [WxH] mask tensor
             of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A float [4x4] initial
             transformation matrix.

        Returns: (:obj:`torch.Tensor`): A [4x4] rigid motion matrix
            that aligns source points to target points.

        """

        if transform is None:
            transform = torch.eye(4, device=target_points.device)

        for scale, icp_instance in self.scale_icp:
            if scale < 1.0:
                tgt_points = downsample_xyz(target_points, target_mask, scale,
                                            method=self.downsample_xyz_method)
                tgt_normals = downsample_xyz(target_normals, target_mask, scale,
                                             normalize=True,
                                             method=self.downsample_xyz_method)
                tgt_mask = downsample(target_mask, scale,
                                      DownsampleMethod.Nearest)

                src_points = downsample_xyz(source_points, source_mask, scale,
                                            normalize=False,
                                            method=self.downsample_xyz_method)
                src_mask = downsample(
                    source_mask, scale, DownsampleMethod.Nearest)
            else:
                tgt_points = target_points
                tgt_normals = target_normals
                tgt_mask = target_mask

                src_points = source_points
                src_mask = source_mask

            scaled_kcam = kcam.scaled(scale)
            transform = icp_instance.estimate(tgt_points, tgt_normals, tgt_mask,
                                              src_points, src_mask, scaled_kcam,
                                              transform)

        return transform

    def estimate_frame_to_frame(self, target_frame, source_frame, transform=None):
        return self.estimate(target_frame.points, target_frame.normals,
                             target_frame.mask, source_frame.points, source_frame.mask,
                             source_frame.kcam, transform)
