"""Pose estimation via iterative closest points algorithm.
"""
import math

import torch
from tenviz.pose import SE3, SO3

from fiontb.downsample import DownsampleXYZMethod
from fiontb._cfiontb import ICPJacobian as _ICPJacobian
from fiontb._utils import empty_ensured_size

from .result import ICPResult, ICPVerifier
from .multiscale_optim import MultiscaleOptimization as _MultiscaleOptimization

# pylint: disable=invalid-name


class _Step:
    def __init__(self, geom, so3=False):
        self.JtJ = None
        self.Jtr = None
        self.residual = None
        self.geom = geom
        self.so3 = so3

    def __call__(self, target_points, target_normals, target_feats, target_mask,
                 source_points, source_feats, source_mask, kcam, transform):
        num_params = 6
        if self.so3:
            num_params = 3
        device = target_points.device
        dtype = source_points.dtype

        self.JtJ = empty_ensured_size(
            self.JtJ, source_points.size(0), num_params, num_params,
            device=device, dtype=dtype)
        self.Jtr = empty_ensured_size(
            self.Jtr, source_points.size(0), num_params,
            device=device, dtype=dtype)
        self.residual = empty_ensured_size(
            self.residual, source_points.size(0),
            device=device, dtype=dtype)

        if not self.so3:
            if self.geom:
                match_count = _ICPJacobian.estimate_geometric(
                    target_points, target_normals, target_mask,
                    source_points, source_mask, kcam, transform,
                    self.JtJ, self.Jtr, self.residual)
            else:
                match_count = _ICPJacobian.estimate_feature(
                    target_points, target_normals, target_feats,
                    target_mask, source_points, source_feats,
                    source_mask, kcam, transform,
                    self.JtJ, self.Jtr, self.residual)
        else:
            match_count = _ICPJacobian.estimate_feature_so3(
                target_points, target_normals, target_feats,
                target_mask, source_points, source_feats,
                source_mask, kcam, transform,
                self.JtJ, self.Jtr, self.residual)

        Jtr = self.Jtr.sum(0)
        JtJ = self.JtJ.sum(0)
        residual = self.residual.sum()

        return JtJ, Jtr, residual, match_count


class ICPOdometry:
    """Point-to-plane iterative closest points
    algorithm.

    Attributes:

        num_iters (int): Number of iterations for the Gauss-Newton
         optimization algorithm.

    """

    def __init__(self, num_iters, geom_weight=1.0, feat_weight=1.0, so3=False):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3

        self._geom_step = _Step(
            True, so3=so3) if geom_weight > 0 and not so3 else None
        self._feature_step = _Step(False, so3=so3) if feat_weight > 0 else None

    def estimate(self, kcam, source_points, source_mask, source_feats=None,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, transform=None):
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

        geom_only = target_feats is None or source_feats is None

        if not geom_only:
            source_feats = source_feats.view(source_feats.size(0), -1)

        best_result = ICPResult(None, None, math.inf, 0)

        has_features = (source_feats is not None
                        and target_feats is not None
                        and self._feature_step is not None)

        for _ in range(self.num_iters):
            JtJ = torch.zeros(6, 6, dtype=torch.double)
            Jr = torch.zeros(6, dtype=torch.double)
            residual = 0
            match_count = 0

            if has_features:
                feat_JtJ, feat_Jr, feat_residual, feat_count = self._feature_step(
                    target_points, target_normals, target_feats,
                    target_mask, source_points, source_feats,
                    source_mask, kcam, transform)
                JtJ = feat_JtJ.cpu().double()*self.feat_weight*self.feat_weight
                Jr = -feat_Jr.cpu().double()*self.feat_weight
                residual = feat_residual*self.feat_weight
                match_count = feat_count

            if self._geom_step is not None:
                geom_JtJ, geom_Jr, geom_residual, geom_count = self._geom_step(
                    target_points, target_normals, None, target_mask,
                    source_points, None, source_mask, kcam, transform)

                JtJ += geom_JtJ.cpu().double()*self.geom_weight*self.geom_weight
                Jr += geom_Jr.cpu().double()*self.geom_weight
                residual += geom_residual*self.geom_weight
                match_count = max(geom_count, match_count)

            try:
                # update = JtJ.cpu().inverse() @ Jr.cpu()

                upper_JtJ = torch.cholesky(JtJ)
                Jr = Jr.cpu().view(-1, 1).double()
                update = torch.cholesky_solve(
                    Jr, upper_JtJ).squeeze()
            except:
                break

            if not self.so3:
                update_matrix = SE3.exp(
                    update).to_matrix().to(device).to(dtype)
                transform = update_matrix @ transform
            else:
                update_matrix = SO3.exp(
                    update).to_matrix().to(device).to(dtype)
                transform = update_matrix @ transform

            residual = residual.item() / match_count

            best_result = ICPResult(
                transform.cpu(), JtJ, residual, match_count / source_points.size(0))

            # Uncomment the next line(s) for debug
            # print(_, residual)

        return best_result

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        from fiontb.frame import FramePointCloud

        source_frame = FramePointCloud.from_frame(source_frame).to(device)
        target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(source_frame.kcam, source_frame.points, source_frame.mask,
                             source_feats=source_feats,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             target_feats=target_feats,
                             transform=transform)


class ICPOption:
    def __init__(self, scale, iters=30, geom_weight=1.0, feat_weight=1.0, so3=False):
        self.scale = scale
        self.iters = iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3


class MultiscaleICPOdometry(_MultiscaleOptimization):
    """Pyramidal point-to-plane iterative closest points
    algorithm.
    """

    def __init__(self, options, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Initialize the multiscale ICP.

        Args:

            scale_iters (List[(float, int, bool)]): Scale levels, its
             number of iterations, and whatever should use features. Scales must be <= 1.0

            downsample_xyz_method
             (:obj:`fiontb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the xyz target and source points.
        """

        super().__init__(
            [(opt.scale, ICPOdometry(
                opt.iters, opt.geom_weight, opt.feat_weight, opt.so3))
             for opt in options],
            downsample_xyz_method)
