"""Pose estimation via iterative closest points algorithm.
"""
import math

import torch
from tenviz.pose import SE3

from fiontb.downsample import DownsampleXYZMethod
from fiontb._cfiontb import ICPJacobian as _ICPJacobian
from fiontb._utils import empty_ensured_size

from .result import ICPResult, ICPVerifier
from .multiscale_optim import MultiscaleOptimization as _MultiscaleOptimization

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
        first_loss = None
        best_match_count = None
        best_JtJ = None

        for _ in range(self.num_iters):
            if geom_only:
                match_count = _ICPJacobian.estimate_geometric(
                    target_points, target_normals, target_mask,
                    source_points, source_mask, kcam, transform,
                    self.jacobian, self.residual)
            else:
                match_count = _ICPJacobian.estimate_hybrid(
                    target_points, target_normals, target_feats,
                    target_mask, source_points, source_feats,
                    source_mask, kcam, transform, geom_weight, feat_weight,
                    self.jacobian, self.residual)

            Jt = self.jacobian.transpose(1, 0)
            JtJ = Jt @ self.jacobian

            Jr = Jt @ self.residual

            JtJ = JtJ.cpu().double()
            Jr = Jr.view(-1, 1).cpu().double()

            try:
                # update = JtJ.cpu().inverse() @ Jr.cpu()
                upper_JtJ = torch.cholesky(JtJ)

                update = torch.cholesky_solve(
                    Jr, upper_JtJ).squeeze()
            except:
                best_loss = math.inf
                break

            update_matrix = SE3.exp(
                update).to_matrix().to(device).to(dtype)
            transform = update_matrix @ transform

            loss = abs(self.residual.mean().item())

            if first_loss is None:
                first_loss = loss

            if loss < best_loss:
                best_loss = loss
                best_transform = transform
                best_match_count = match_count
                best_JtJ = JtJ

            # Uncomment the next line(s) for optimization debug
            #print(_, loss)

        #print(first_loss, best_loss)

        return ICPResult(best_transform, best_JtJ,
                         loss, best_loss, best_match_count /
                         source_points.size(0),
                         match_count / source_points.size(0))

    def estimate_frame(self, source_frame, target_frame,
                       transform=None, geom_weight=1.0, feat_weight=1.0,
                       device="cpu"):
        from fiontb.frame import FramePointCloud

        source_frame = FramePointCloud.from_frame(source_frame).to(device)
        target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(source_frame.kcam, source_frame.points, source_frame.mask,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             transform=transform,
                             geom_weight=geom_weight, feat_weight=feat_weight)


class MultiscaleICPOdometry(_MultiscaleOptimization):
    """Pyramidal point-to-plane iterative closest points
    algorithm.
    """

    def __init__(self, scale_iters, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Initialize the multiscale ICP.

        Args:

            scale_iters (List[(float, int, bool)]): Scale levels, its
             number of iterations, and whatever should use features. Scales must be <= 1.0

            downsample_xyz_method
             (:obj:`fiontb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the xyz target and source points.
        """

        super().__init__(
            [(scale, ICPOdometry(iters), use_feats)
             for scale, iters, use_feats in scale_iters],
            downsample_xyz_method)
