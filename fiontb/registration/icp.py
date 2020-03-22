"""Pose estimation via iterative closest points algorithm.
"""
import math

import torch
from tenviz.pose import SE3, SO3

from fiontb.processing import DownsampleXYZMethod
from fiontb.frame import FramePointCloud, Frame
from fiontb._cfiontb import ICPJacobian as _ICPJacobian
from fiontb._utils import empty_ensured_size

from .result import ICPResult, ICPVerifier
from .multiscale_optim import MultiscaleOptimization as _MultiscaleOptimization

# pylint: disable=invalid-name


class _Step:
    def __init__(self, geom, so3, distance_threshold, normals_angle_thresh, feat_residual_thresh):
        self.JtJ = None
        self.Jtr = None
        self.residual = None
        self.geom = geom
        self.so3 = so3
        self.distance_threshold = distance_threshold
        self.normals_angle_thresh = normals_angle_thresh
        self.feat_residual_thresh = feat_residual_thresh

    def __call__(self, target_points, target_normals, target_feats, target_mask,
                 source_points, source_normals, source_feats, source_mask,
                 kcam, transform):
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
                    source_points, source_normals, source_mask,
                    kcam, transform.to(dtype),
                    self.distance_threshold, self.normals_angle_thresh,
                    self.JtJ, self.Jtr, self.residual)
            else:
                match_count = _ICPJacobian.estimate_feature(
                    target_points, target_normals, target_feats,
                    target_mask, source_points,
                    source_feats, source_mask, kcam, transform.to(dtype),
                    self.distance_threshold, self.normals_angle_thresh, self.feat_residual_thresh,
                    self.JtJ, self.Jtr, self.residual)
        else:
            match_count = _ICPJacobian.estimate_feature_so3(
                target_points, target_normals, target_feats,
                target_mask, source_points, source_feats,
                source_mask, kcam, transform.to(dtype),
                self.distance_threshold, self.normals_angle_thresh, self.feat_residual_thresh,
                self.JtJ, self.Jtr, self.residual)

        Jtr = self.Jtr.double().sum(0)
        JtJ = self.JtJ.double().sum(0)
        residual = self.residual.sum()

        return JtJ, Jtr, residual, match_count


class ICPOdometry:
    """Iterative closest points algorithm using the point-to-plane error,
    and Gauss-Newton optimization procedure.

    Attributes:

        num_iters (int): Number of iterations for the Gauss-Newton
         optimization algorithm.

        geom_weight (float): Geometry term weighting, 0.0 to disable
         use of depth data.

        feat_weight (float): Feature term weighting, 0.0 to ignore
         point features.

        so3 (bool): SO3 optimization, i.e., rotation only.

        distance_threshold (float): Maximum distance to match a pair
         of source and target points.

        normals_angle_thresh (float): Maximum angle in radians between
         normals to match a pair of source and target points.

        feat_residual_thresh (float): Maximum residual between features.
    """

    def __init__(self, num_iters, geom_weight=1.0, feat_weight=1.0, so3=False,
                 distance_threshold=0.1, normals_angle_thresh=math.pi/8.0, feat_residual_thresh=0.5):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3

        self._geom_step = (_Step(True, so3, distance_threshold,
                                 normals_angle_thresh, feat_residual_thresh)
                           if geom_weight > 0 and not so3 else None)
        self._feature_step = (_Step(False, so3, distance_threshold,
                                    normals_angle_thresh, feat_residual_thresh)
                              if feat_weight > 0 else None)

    def estimate(self, kcam, source_points, source_normals,
                 source_mask, source_feats=None,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, transform=None):
        """Estimate the odometry between a target points and normals in a
        grid against source points using the point-to-plane geometric
        error.

        Args:

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            source_points (:obj:`torch.Tensor`): A float (N x 3) tensor of
             source points.

            source_normals (:obj:`torch.Tensor`): A float (N x 3) tensor of
             source normals.

            source_mask (:obj:`torch.Tensor`): A bool (N) mask
             tensor of valid source points.

            target_points (:obj:`torch.Tensor`): A float (H x W x 3) tensor
             of rasterized 3d points that the source points should be
             aligned with.

            target_normals (:obj:`torch.Tensor`): A float (H x W x 3) tensor
             of rasterized 3d **normals** that the source points should be
             aligned with.

            target_mask (:obj:`torch.Tensor`): A bool (H x W) mask
              tensor of valid target points.

            transform (:obj:`torch.Tensor`): A float (4 x 4) initial
             transformation matrix.

        Returns:

            (:obj:`ICPResult`): Resulting transformation and information.
        """

        device = target_points.device
        kcam = kcam.matrix.to(device)
        if transform is None:
            transform = torch.eye(4, device=device, dtype=torch.double)
        else:
            transform = transform.to(device).double()

        source_points = source_points.view(-1, 3)
        source_normals = source_normals.view(-1, 3)
        source_mask = source_mask.view(-1)

        geom_only = target_feats is None or source_feats is None

        if not geom_only:
            source_feats = source_feats.view(source_feats.size(0), -1)

        best_result = ICPResult()

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
                    target_mask, source_points, source_normals, source_feats,
                    source_mask, kcam, transform)
                JtJ = feat_JtJ.cpu().double()*self.feat_weight*self.feat_weight
                Jr = -feat_Jr.cpu().double()*self.feat_weight
                residual = feat_residual*self.feat_weight
                match_count = feat_count

            if self._geom_step is not None:
                geom_JtJ, geom_Jr, geom_residual, geom_count = self._geom_step(
                    target_points, target_normals, None, target_mask,
                    source_points, source_normals, None, source_mask,
                    kcam, transform)

                JtJ += geom_JtJ.cpu().double()*self.geom_weight*self.geom_weight
                Jr += geom_Jr.cpu().double()*self.geom_weight
                residual += geom_residual*self.geom_weight
                match_count = max(geom_count, match_count)

            try:
                upper_JtJ = torch.cholesky(JtJ)
                Jr = Jr.cpu().view(-1, 1).double()
                update = torch.cholesky_solve(
                    Jr, upper_JtJ).squeeze()
            except:
                break

            if not self.so3:
                update_matrix = SE3.exp(
                    update).to_matrix().to(device)
                transform = update_matrix @ transform
            else:
                update_matrix = SO3.exp(
                    update).to_matrix().to(device)
                transform = update_matrix @ transform

            residual = residual.item() / match_count
            best_result = ICPResult(
                transform.cpu(), JtJ, residual, match_count / source_points.size(0))

        return best_result

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        """Estimate the odometry between two frames.

        Args:

            source_frame (Union[:obj:`fiontb.frame.Frame`,
             :obj:`fiontb.frame.FramePointCloud`]): Source frame.

            target_frame (Union[:obj:`fiontb.frame.Frame`,
             :obj:`fiontb.frame.FramePointCloud`]): Target frame.

            source_feats (:obj:`torch.Tensor`, optional): Source
             feature map (C x H x W).

            target_feats (:obj:`torch.Tensor`, optional): Target
             feature map (C x H x W).

            transform (:obj:`torch.Tensor`, optional): Initial
             transformation, (4 x 4) matrix.

            device (str): Torch device to execute the algorithm.

        Returns:

            (:obj:`ICPResult`): Resulting transformation and
             information.

        """

        if isinstance(source_frame, Frame):
            source_frame = FramePointCloud.from_frame(source_frame).to(device)

        if isinstance(target_frame, Frame):
            target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(source_frame.kcam, source_frame.points,
                             source_frame.normals,
                             source_frame.mask,
                             source_feats=source_feats,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             target_feats=target_feats,
                             transform=transform)


class ICPOptions:
    """Options for the ICP algorithm.

    Attributes:

        scale (float): Resizing scale for inputs.

        iters (int): Number of optimizer iterations.

        geom_weight (float): Geometry term weighting, 0.0 to disable
         use of depth data.

        feat_weight (float): Feature term weighting, 0.0 to ignore
         point features.

        so3 (bool): SO3 optimization, i.e., rotation only.

        distance_threshold (float): Maximum distance to match a pair
         of source and target points.

        normals_angle_thresh (float): Maximum angle in radians between
         normals to match a pair of source and target points.

    """

    def __init__(self, scale, iters=30, geom_weight=1.0, feat_weight=1.0, so3=False,
                 distance_threshold=0.1, normals_angle_thresh=math.pi/2.0):
        self.scale = scale
        self.iters = iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3
        self.distance_threshold = distance_threshold
        self.normals_angle_thresh = normals_angle_thresh


class MultiscaleICPOdometry(_MultiscaleOptimization):
    """Refines point-to-plane ICP by leveraging from lower resolution results.
    """

    def __init__(self, options, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Initialize the multiscale ICP.

        Args:

            options (List[:obj:`ICPOption`]): Each element contains a
             scale specific ICP options. Options should be specified
             with their scales from higher to lower. And they're applied
             from lower to higher.

            downsample_xyz_method
             (:obj:`fiontb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the XYZ points and normals.

        """

        super().__init__(
            [(opt.scale, ICPOdometry(
                opt.iters, opt.geom_weight, opt.feat_weight, opt.so3))
             for opt in options],
            downsample_xyz_method)
