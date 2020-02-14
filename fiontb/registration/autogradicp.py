"""Pose estimation using PyTorch's Autograd functionality.
"""
import math

import torch
import scipy.optimize

from fiontb.frame import Frame, FramePointCloud
from fiontb.camera import RigidTransform
from fiontb.spatial.matching import (FramePointCloudMatcher,
                                     PointCloudMatcher)
from fiontb.processing import DownsampleXYZMethod

from .so3 import SO3tExp
from .result import ICPResult
from .multiscale_optim import MultiscaleOptimization as _MultiscaleOptimization


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
            self.x[0][0] = float(x[0])
            self.x[0][1] = float(x[1])
            self.x[0][2] = float(x[2])
            self.x[0][3] = float(x[3])
            self.x[0][4] = float(x[4])
            self.x[0][5] = float(x[5])
        loss = self.closure()

        if loss == 0:
            return 0
        value = loss.detach().cpu().item()

        loss.backward()
        grad = self.x.grad.cpu().numpy()
        grad = grad.transpose().flatten()
        return value, grad


class AutogradICP:
    """ICP using PyTorch's autograd function for estimating 6D
    gradient. The pose is optimized by the BFGS algorithm.

    Attributes:

        num_iters (int): Number of iterations for the BFGS
         optimization algorithm.

        learning_rate (float): Scaling factor for the cost
         function. Not rarely, BFGS will try high value parameter
         values when estimating the Hessian matrix. If they're high
         enough, it'll cause lookups too far outside the target search
         bounds, causing errors. On our tests good values are 0.01
         between and 0.1.

        geom_weight (float): Geometry term weighting, 0.0 to disable
         use of depth data.

        feat_weight (float): Feature term weighting, 0.0 to ignore
         point features.
    """

    def __init__(self, num_iters, learning_rate=0.01,
                 geom_weight=0.5, feat_weight=0.5):
        self.num_iters = num_iters
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.learning_rate = learning_rate

    def _estimate(self, source_points, source_normals,
                  source_feats, target_matcher, transform=None):
        device = source_points.device
        dtype = source_points.dtype

        upsilon_omega = torch.zeros(
            1, 6, requires_grad=True, device=device, dtype=dtype)
        initial_transform = None
        if transform is not None:
            # TODO: investigate non-orthogonal matrices erro
            # se3 = SE3.from_matrix(transform.cpu())
            # upsilon_omega = se3.log().to(device).to(dtype).view(1, 6)
            # upsilon_omega.requires_grad = True
            source_points = RigidTransform(
                transform.to(device)) @ source_points.clone()
            initial_transform = transform.clone()

        loss = math.inf

        has_geom = self.geom_weight > 0
        has_feat = (self.feat_weight > 0
                    and source_feats is not None)

        def _closure():
            nonlocal loss

            if upsilon_omega.grad is not None:
                upsilon_omega.grad.zero_()

            transform = SO3tExp.apply(upsilon_omega.cpu()).squeeze().to(device)

            geom_loss = 0
            feat_loss = 0

            transform_source_points = (
                RigidTransform(transform) @ source_points)
            tpoints, tnormals, tfeatures, smask = target_matcher.find_correspondences(
                transform_source_points)

            snormals = source_normals[smask]
            normal_mask = torch.bmm(tnormals.view(-1, 1, 3),
                                    snormals.view(-1, 3, 1)).squeeze() > 0.5

            feat_diff = torch.norm(tfeatures - source_feats[:, smask], 2, dim=0)
            #feat_diff = -torch.cosine_similarity(tfeatures, source_feats[:, smask], dim=0)
            
            spoints = transform_source_points[smask]

            #mmask = normal_mask & (feat_diff < 0.2)
            #mmask = feat_diff < 0.5
            mmask = torch.ones(spoints.size(0), dtype=torch.bool)

            if has_geom:
                diff = tpoints[mmask] - spoints[mmask]

                cost = torch.bmm(tnormals[mmask].view(
                    -1, 1, 3), diff.view(-1, 3, 1))
                geom_loss = torch.pow(cost, 2).mean()

            if has_feat:
                feat_loss = feat_diff[mmask].mean()

            loss = geom_loss*self.geom_weight + feat_loss*self.feat_weight

            if torch.isnan(loss):
                return loss

            return loss*self.learning_rate

        box = _ClosureBox(_closure, upsilon_omega)
        scipy.optimize.minimize(box.func, upsilon_omega.detach().cpu().numpy(),
                                method='BFGS', jac=True,
                                options=dict(disp=False, maxiter=self.num_iters))
        transform = SO3tExp.apply(upsilon_omega.detach().cpu()).squeeze(0)
        transform = _to_4x4(transform)

        if initial_transform is not None:
            transform = _to_4x4(initial_transform) @ transform

        return ICPResult(transform, None, loss, 1.0)

    def estimate(self, kcam, source_points, source_normals, source_mask,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, source_feats=None, transform=None):
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

            (:obj:`fiontb.registration.result.ICPResult`): Resulting transformation and alignment information.
        """

        source_points = source_points[source_mask].view(-1, 3)
        source_normals = source_normals[source_mask].view(-1, 3)

        if source_feats is not None:
            source_feats = source_feats[:, source_mask].view(
                source_feats.size(0), -1)

        matcher = FramePointCloudMatcher(target_points, target_normals, target_mask,
                                         target_feats, kcam)

        return self._estimate(source_points, source_normals,
                              source_feats, matcher, transform)

    def estimate_pcl(self, source_pcl, target_pcl, transform=None, source_feats=None,
                     target_feats=None):
        """Estimate the alignment between two point clouds.

        Args:

            source_pcl (Union[:obj:`fiontb.pointcloud.PointCloud`,
             :obj:`fiontb.surfel.SurfelCloud`]): Source point cloud.

            target_pcl (Union[:obj:`fiontb.pointcloud.PointCloud`,
             :obj:`fiontb.surfel.SurfelCloud`]): Target pcl.

            source_feats (:obj:`torch.Tensor`, optional): Source
             feature map (C x N).

            target_feats (:obj:`torch.Tensor`, optional): Target
             feature map (C x N).

            transform (:obj:`torch.Tensor`, optional): Initial
             transformation, (4 x 4) matrix.

        Returns:

            (:obj:`fiontb.registration.result.ICPResult`): Resulting transformation and
             alignment information.

        """

        matcher = PointCloudMatcher(
            target_pcl.points, target_pcl.normals, target_feats,
            num_neighbors=8, distance_upper_bound=0.5)
        return self._estimate(source_pcl.points, source_pcl.normals,
                              source_feats, matcher, transform)

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

            (:obj:`fiontb.registration.result.ICPResult`): Resulting transformation and
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


class AGICPOption:
    """Options for the AutogradICP algorithm.

    Attributes:

        scale (float): Resizing scale for inputs.

        iters (int): Number of optimizer iterations.

        geom_weight (float): Geometry term weighting, 0.0 to disable
         use of depth data.

        feat_weight (float): Feature term weighting, 0.0 to ignore
         point features.

        learning_rate (float): Scaling factor for the cost
         function. Not rarely, BFGS will try high value parameter
         values when estimating the Hessian matrix. If they're high
         enough, it'll cause lookups too far outside the target search
         bounds, causing errors. On our tests good values are 0.01
         between and 0.1.
    """

    def __init__(self, scale, iters=30, learning_rate=5e-2,
                 geom_weight=1.0, feat_weight=1.0, so3=False):
        self.scale = scale
        self.iters = iters
        self.learning_rate = learning_rate
        self.geom_weight = geom_weight
        self.feat_weight = feat_weight
        self.so3 = so3


class MultiscaleAutogradICP(_MultiscaleOptimization):
    """Refines point-to-plane AutogradICP by leveraging from lower
    resolution results.

    """

    def __init__(self, options, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        super().__init__(
            [(opt.scale, AutogradICP(
                opt.iters, opt.learning_rate, opt.geom_weight,
                opt.feat_weight))
             for opt in options],
            downsample_xyz_method)

    def estimate_pcl(self, source_pcl, target_pcl, transform=None,
                     device="cpu"):
        """Multiscale AutogradICP for point clouds.
        """

        for scale, estimator in self.estimators:
            src = source_pcl.to("cpu").downsample(scale).to(device)
            tgt = target_pcl.to("cpu").downsample(scale).to(device)

            result = estimator.estimate_pcl(src, tgt, transform=transform.to(device),
                                            source_feats=src.features,
                                            target_feats=tgt.features)
            transform = result.transform

        return result
