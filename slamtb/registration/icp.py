"""Pose estimation via iterative closest points algorithm.
"""
import math

import torch
from tenviz.pose import SE3, SO3

from slamtb.frame import FramePointCloud, Frame
from .correspondence_map import CorrespondenceMap
from slamtb._cslamtb import ICPJacobian as _ICPJacobian

from slamtb._utils import empty_ensured_size

from .result import RegistrationResult


# pylint: disable=invalid-name


class _JacobianStep:
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
                corresp_map_func = CorrespondenceMap(self.distance_threshold,
                                                     self.normals_angle_thresh)
                corresp_map = corresp_map_func(
                    source_points, source_normals, source_mask,
                    transform.to(dtype),
                    target_points, target_normals, target_mask,
                    kcam)

                match_count = _ICPJacobian.estimate_feature(
                    source_points, source_feats, source_mask, transform.to(
                        dtype),
                    target_feats, kcam, corresp_map, self.feat_residual_thresh,
                    self.JtJ, self.Jtr, self.residual)
                print(match_count)
        else:
            match_count = _ICPJacobian.estimate_feature_so3(
                target_points, target_normals, target_feats,
                target_mask, source_points, source_normals,
                source_feats, source_mask, kcam, transform.to(dtype),
                self.distance_threshold, self.normals_angle_thresh, self.feat_residual_thresh,
                self.JtJ, self.Jtr, self.residual)

        Jtr = self.Jtr.sum(0)
        JtJ = self.JtJ.sum(0)
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
                 distance_threshold=.5, normals_angle_thresh=math.pi/4,
                 feat_residual_thresh=2.75):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3

        self._geom_step = (_JacobianStep(True, so3, distance_threshold,
                                         normals_angle_thresh, feat_residual_thresh)
                           if geom_weight > 0 and not so3 else None)
        self._feature_step = (_JacobianStep(False, so3, distance_threshold,
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

            kcam (:obj:`slamtb.camera.KCamera`): Intrinsics camera
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

            (:obj:`RegistrationResult`): Resulting transformation and information.
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

        has_features = not None in (source_feats, target_feats,
                                    self._feature_step)
        if has_features:
            source_feats = source_feats.view(source_feats.size(0), -1)

        best_result = RegistrationResult()
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
            best_result = RegistrationResult(
                transform.cpu(), JtJ, residual, match_count / source_points.size(0))

        return best_result

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        """Estimate the odometry between two frames.

        Args:

            source_frame (Union[:obj:`slamtb.frame.Frame`,
             :obj:`slamtb.frame.FramePointCloud`]): Source frame.

            target_frame (Union[:obj:`slamtb.frame.Frame`,
             :obj:`slamtb.frame.FramePointCloud`]): Target frame.

            source_feats (:obj:`torch.Tensor`, optional): Source
             feature map (C x H x W).

            target_feats (:obj:`torch.Tensor`, optional): Target
             feature map (C x H x W).

            transform (:obj:`torch.Tensor`, optional): Initial
             transformation, (4 x 4) matrix.

            device (str): Torch device to execute the algorithm.

        Returns:

            (:obj:`RegistrationResult`): Resulting transformation and
             information.

        """

        if isinstance(source_frame, Frame):
            source_frame = FramePointCloud.from_frame(source_frame).to(device)

        if isinstance(target_frame, Frame):
            target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(
            source_frame.kcam,
            source_frame.points,
            source_frame.normals,
            source_frame.mask,
            source_feats=source_feats,
            target_points=target_frame.points,
            target_normals=target_frame.normals,
            target_mask=target_frame.mask,
            target_feats=target_feats,
            transform=transform)
