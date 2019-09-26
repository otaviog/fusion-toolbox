"""Pose estimation via iterative closest points algorithm.
"""
import math

import torch
from tenviz.pose import SE3

from fiontb.downsample import (
    downsample_xyz, downsample_mask, DownsampleXYZMethod, DownsampleMethod)
from fiontb._cfiontb import (ICPJacobian as _ICPJacobian,
                             calc_sobel_gradient)
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

    def estimate(self, kcam, source_points, source_mask, source_feats=None,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, transform=None,
                 geom_weight=1.0, feat_weight=1.0):
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
        # assert source_points.is_cuda, "At least source_points must be a cuda tensor."

        device = target_points.device
        dtype = source_points.dtype
        kcam = kcam.matrix.to(device)
        if transform is None:
            transform = torch.eye(4, device=device, dtype=dtype)
        else:
            transform = transform.to(device)

        source_points = source_points.view(-1, 3)
        source_mask = source_mask.view(-1)

        self.jacobian = empty_ensured_size(self.jacobian, source_points.size(0), 6,
                                           device=device, dtype=dtype)
        self.residual = empty_ensured_size(self.residual, source_points.size(0),
                                           device=device, dtype=dtype)

        geom_only = target_feats is None or source_feats is None

        if not geom_only:
            source_feats = source_feats.view(-1, source_points.size(0))

        best_loss = math.inf
        best_transform = None

        for _ in range(self.num_iters):
            if geom_only:
                _ICPJacobian.estimate_geometric(
                    target_points, target_normals, target_mask,
                    source_points, source_mask, kcam, transform,
                    self.jacobian, self.residual)
            else:
                _ICPJacobian.estimate_hybrid(
                    target_points, target_normals, target_feats,
                    target_mask, source_points, source_feats,
                    source_mask, kcam, transform, geom_weight, feat_weight,
                    self.jacobian, self.residual)

            Jt = self.jacobian.transpose(1, 0)
            JtJ = Jt @ self.jacobian
            Jr = Jt @ self.residual

            if True:
                upper_JtJ = torch.cholesky(JtJ.double())

                update = torch.cholesky_solve(
                    Jr.view(-1, 1).double(), upper_JtJ).squeeze()
            else:
                update = JtJ.inverse() @ Jr

            update_matrix = SE3.exp(
                update.cpu()).to_matrix().to(device).to(dtype)
            transform = update_matrix @ transform

            # loss = torch.pow(self.residual, 2).mean().item()
            loss = self.residual.mean().item()

            if loss < best_loss:
                best_loss = loss
                best_transform = transform

            # Uncomment the next line(s) for optimization debug
            print(_, loss)

        return best_transform

    def estimate_frame_to_frame(self, source_frame, target_frame,
                                transform=None, geom_weight=1.0, feat_weight=1.0):
        return self.estimate(source_frame.kcam, source_frame.points, source_frame.mask,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             transform=transform,
                             geom_weight=geom_weight, feat_weight=feat_weight)


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

    def estimate(self, kcam, source_points, source_mask,
                 target_points, target_mask, target_normals,
                 transform=None):
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

        Returns: (:obj:`torch.Tensor`): A [3x4] rigid motion matrix
            that aligns source points to target points.

        """

        if transform is None:
            transform = torch.eye(4, device=source_points.device,
                                  dtype=source_points.dtype)

        for scale, icp_instance in self.scale_icp:
            if scale < 1.0:
                tgt_points = downsample_xyz(target_points, target_mask, scale,
                                            method=self.downsample_xyz_method)
                tgt_normals = downsample_xyz(target_normals, target_mask, scale,
                                             normalize=True,
                                             method=self.downsample_xyz_method)

                tgt_mask = downsample_mask(target_mask, scale)

                src_points = downsample_xyz(source_points, source_mask, scale,
                                            normalize=False,
                                            method=self.downsample_xyz_method)
                src_mask = downsample_mask(source_mask, scale)
            else:
                tgt_points = target_points
                tgt_normals = target_normals
                tgt_mask = target_mask

                src_points = source_points
                src_mask = source_mask

            scaled_kcam = kcam.scaled(scale)
            transform = icp_instance.estimate(
                scaled_kcam, src_points, src_mask,
                target_points=tgt_points,
                target_mask=tgt_mask, target_normals=tgt_normals,
                transform=transform, geom_weight=1.0, feat_weight=0)

        return transform

    def estimate_frame_to_frame(self, target_frame, source_frame, transform=None):
        return self.estimate(target_frame.points, target_frame.normals,
                             target_frame.mask, source_frame.points, source_frame.mask,
                             source_frame.kcam, transform)


class MultiscaleFeatICPOdometry:
    def __init__(self, scale_iters):
        pass

    def estimate(self,
                 kcam, source_points, source_mask,
                 target_points, target_mask, target_normals,
                 transform=None):
        pass
