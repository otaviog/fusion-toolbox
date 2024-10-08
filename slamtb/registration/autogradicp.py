"""Pose estimation using PyTorch's Autograd functionality.
"""

import math

import torch
import torch.optim
import scipy.optimize

from slamtb.frame import Frame, FramePointCloud
from slamtb.camera import RigidTransform
from slamtb.spatial.matching import (FramePointCloudMatcher,
                                     PointCloudMatcher)

from .se3 import ExpRtToMatrix, MatrixToExpRt
from .result import RegistrationResult


def _to_4x4(mtx):
    if mtx.size(0) == 4 and mtx.size(1) == 4:
        return mtx

    ret = torch.eye(4, device=mtx.device, dtype=mtx.dtype)
    ret[:3, :4] = mtx[:3, :4]

    return ret


class _ClosureBox:
    def __init__(self, closure, x):
        self.closure = closure
        self.x = x

    def func(self, x):
        """Evaluates at x (6D).

        Returns:

            (float, List[float]): The cost value and 6D gradient.
        """

        with torch.no_grad():
            self.x[0] = float(x[0])
            self.x[1] = float(x[1])
            self.x[2] = float(x[2])
            self.x[3] = float(x[3])
            self.x[4] = float(x[4])
            self.x[5] = float(x[5])
        loss = self.closure()

        if loss == 0:
            return 0
        value = loss.detach().cpu().item()

        loss.backward()
        grad = self.x.grad.cpu().numpy()

        return value, grad


class _HuberLoss(torch.nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, alpha):
        """
        Args:

            alpha (float): Huber loss alpha value.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, residual):
        """Module forward
        """
        loss1 = residual[residual < self.alpha]
        #loss1 = torch.pow(loss1, 2)

        loss2 = residual[residual >= self.alpha]
        loss2 = (self.alpha*torch.abs(loss2)
                 - self.alpha)

        loss = 0

        if loss1.numel() > 0:
            loss = loss1.mean()

        if loss2.numel() > 0:
            loss += loss2.mean()

        return loss

# class Optimizer(Enum):


class AutogradICP:
    """ICP using PyTorch's autograd for estimating 6D gradient
    (translation + exponential map). The pose is optimized by the BFGS
    algorithm.

    Attributes:

        num_iters (int): Number of iterations for the BFGS
         optimization algorithm.

        learning_rate (float): Scaling factor for the cost
         function. Not rarely, BFGS will try high parameter values
         when estimating the Hessian matrix. If they're high enough,
         it'll cause lookups outside the target search bounds, causing
         errors. On our tests good values are 0.01 between and 0.1.

        geom_weight (float): Geometry term weighting, 0.0 to disable
         use of depth data.

        feat_weight (float): Feature term weighting, 0.0 to ignore
         point features.

        distance_threshold (float): Maximum distance to match a pair
         of source and target points.

        normals_angle_thresh (float): Maximum angle in radians between
         normals to match a pair of source and target points.

        feat_residual_thresh (float): Maximum residual between features.

        huber_loss_alpha (float): Alpha value for the Huber Loss.
    """

    def __init__(self, num_iters, learning_rate=0.01,
                 geom_weight=0.5, feat_weight=0.5,
                 distance_threshold=0.1,
                 normals_angle_thresh=math.pi/8.0,
                 feat_residual_thresh=0.5,
                 huber_loss_alpha=4.5,
                 gradient_descent_iters=100):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.learning_rate = learning_rate
        self.distance_threshold = distance_threshold
        self.normals_angle_thresh = normals_angle_thresh
        self.feat_residual_thresh = feat_residual_thresh
        self.huber_loss_alpha = huber_loss_alpha
        self.gradient_descent_iters = gradient_descent_iters

    def _estimate(self, source_points, source_normals,
                  source_feats, target_matcher, transform=None):
        device = source_points.device
        dtype = source_points.dtype

        if transform is None:
            exp_rt = torch.zeros(
                6, requires_grad=True, device=device,
                dtype=dtype)
        else:
            exp_rt = MatrixToExpRt.apply(
                transform[:3, :4]).to(device).to(dtype)
            exp_rt.requires_grad = True

        loss = math.inf

        has_geom = self.geom_weight > 0
        has_feat = (self.feat_weight > 0
                    and source_feats is not None)

        huber_loss = _HuberLoss(self.huber_loss_alpha)

        def _closure():
            nonlocal loss
            nonlocal exp_rt
            if exp_rt.grad is not None:
                exp_rt.grad.zero_()
            geom_loss = 0
            feat_loss = 0

            transform = ExpRtToMatrix.apply(exp_rt.cpu()).squeeze().to(device)
            rigid_transform = RigidTransform(transform)
            transform_source_points = rigid_transform @ source_points

            tpoints, tnormals, tfeatures, smask = target_matcher.find_correspondences(
                transform_source_points,
                rigid_transform.transform_normals(source_normals))

            if has_geom:
                diff = tpoints - transform_source_points[smask]
                cost = torch.bmm(tnormals.view(-1, 1, 3),
                                 diff.view(-1, 3, 1)).squeeze()
                geom_loss = torch.pow(cost, 2).mean()

            if has_feat:
                feat_diff = torch.norm(
                    tfeatures - source_feats[:, smask], 2, dim=0)
                if self.feat_residual_thresh > 0:
                    feat_diff = feat_diff[feat_diff.pow(
                        2) < self.feat_residual_thresh*self.feat_residual_thresh]

                if self.huber_loss_alpha > 0:
                    feat_loss = huber_loss(feat_diff)
                else:
                    feat_loss = feat_diff.mean()

            loss = geom_loss*self.geom_weight + feat_loss*self.feat_weight

            if torch.isnan(loss):
                return loss

            return loss*self.learning_rate

        box = _ClosureBox(_closure, exp_rt)
        opt_res = scipy.optimize.minimize(box.func, exp_rt.detach().cpu().numpy(),
                                          method='BFGS', jac=True, tol=0.000000001,
                                          options=dict(disp=False, maxiter=self.num_iters))

        best_exp_rt = exp_rt.detach().clone().cpu()
        best_residual = 5555

        if self.gradient_descent_iters is not None:
            opt = torch.optim.SGD([exp_rt], 0.1)
            opt_res = None
            for _ in range(self.gradient_descent_iters):
                opt.zero_grad()
                _closure()
                loss.backward()

                tmp = loss.cpu().item()

                if tmp < best_residual:
                    best_residual = tmp
                    best_exp_rt = exp_rt.detach().clone().cpu()

                opt.step()

        transform = ExpRtToMatrix.apply(best_exp_rt).squeeze(0)
        transform = _to_4x4(transform)

        return RegistrationResult(
            transform,
            (torch.from_numpy(opt_res.hess_inv).inverse().float()
             if opt_res is not None else None),
            loss, 1.0)

    def estimate(self, kcam, source_points, source_normals, source_mask,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, source_feats=None, transform=None):
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

            (:obj:`slamtb.registration.result.RegistrationResult`): Resulting
             transformation and alignment information.

        """

        source_points = source_points[source_mask].view(-1, 3)
        source_normals = source_normals[source_mask].view(-1, 3)

        if source_feats is not None:
            source_feats = source_feats[:, source_mask].view(
                source_feats.size(0), -1)

        matcher = FramePointCloudMatcher(target_points, target_normals, target_mask,
                                         target_feats, kcam,
                                         self.distance_threshold, self.normals_angle_thresh)

        return self._estimate(source_points, source_normals,
                              source_feats, matcher, transform)

    def estimate_pcl(self, source_pcl, target_pcl, transform=None, source_feats=None,
                     target_feats=None):
        """Estimate the alignment between two point clouds.

        Args:

            source_pcl (Union[:obj:`slamtb.pointcloud.PointCloud`,
             :obj:`slamtb.surfel.SurfelCloud`]): Source point cloud.

            target_pcl (Union[:obj:`slamtb.pointcloud.PointCloud`,
             :obj:`slamtb.surfel.SurfelCloud`]): Target pcl.

            source_feats (:obj:`torch.Tensor`, optional): Source
             feature map (C x N).

            target_feats (:obj:`torch.Tensor`, optional): Target
             feature map (C x N).

            transform (:obj:`torch.Tensor`, optional): Initial
             transformation, (4 x 4) matrix.

        Returns:

            (:obj:`slamtb.registration.result.RegistrationResult`): Resulting transformation and
             alignment information.

        """

        matcher = PointCloudMatcher(
            target_pcl.points, target_pcl.normals, target_feats,
            num_neighbors=8, distance_upper_bound=self.distance_threshold,
            normals_angle_thresh=self.normals_angle_thresh)
        return self._estimate(source_pcl.points, source_pcl.normals,
                              source_feats, matcher, transform)

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

            (:obj:`slamtb.registration.result.RegistrationResult`): Resulting transformation and
             information.

        """
        if isinstance(source_frame, Frame):
            source_frame = FramePointCloud.from_frame(source_frame).to(device)

        if isinstance(target_frame, Frame):
            target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(source_frame.kcam, source_frame.points, source_frame.normals,
                             source_frame.mask,
                             source_feats=source_feats,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             target_feats=target_feats,
                             transform=transform)
