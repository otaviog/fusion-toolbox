import math

from .kdtree_layer import KDTreeLayer
import torch

from fiontb._utils import empty_ensured_size
from fiontb._cfiontb import (FPCLMatcherOp as _FPCLMatcherOp)


class FPCLMatcherOp(torch.autograd.Function):

    class Target:
        def __init__(self, points, normals, mask, features, kcam):
            self.points = points
            self.normals = normals
            self.mask = mask
            self.features = features
            self.kcam = kcam

    class Match:
        def __init__(self):
            self.points = None
            self.normals = None
            self.features = None
            self.mask = None

    @staticmethod
    def forward(ctx, source_points, target, match, grad_precision=0.005):
        device = source_points.device
        dtype = source_points.dtype

        size = source_points.size(0)

        match.points = torch.empty(size, 3, device=device, dtype=dtype)
        match.normals = torch.empty(size, 3, device=device, dtype=dtype)
        match.features = torch.empty(
            target.features.size(0), size, device=device, dtype=dtype)
        match.mask = torch.empty(size, device=device, dtype=torch.bool)

        _FPCLMatcherOp.forward(target.points, target.normals, target.mask,
                               target.features, source_points, target.kcam.matrix.to(device),
                               match.points, match.normals, match.features,
                               match.mask)

        ctx.target = target
        ctx.source_points = source_points
        ctx.match_mask = match.mask
        ctx.grad_precision = grad_precision
        return match.features

    @staticmethod
    def backward(ctx, dl_features):
        device = dl_features.device
        dtype = dl_features.dtype

        dx_points = torch.empty(dl_features.size(1),
                                3, device=device, dtype=dtype)

        _FPCLMatcherOp.backward(ctx.target.features,
                                ctx.source_points,
                                ctx.match_mask,
                                dl_features,
                                ctx.target.kcam.matrix.to(device),
                                ctx.grad_precision,
                                dx_points)

        return dx_points, None, None


class FramePointCloudMatcher:
    def __init__(self, target_points, target_normals, target_mask,
                 target_features, kcam):
        self.target = FPCLMatcherOp.Target(
            target_points, target_normals, target_mask, target_features, kcam)

    @classmethod
    def from_frame_pcl(cls, target_fpcl, features):
        return cls(target_fpcl.points, target_fpcl.normals, target_fpcl.mask,
                   features, target_fpcl.kcam)

    def find_correspondences(self, source_points):
        match = FPCLMatcherOp.Match()
        FPCLMatcherOp.apply(source_points, self.target, match)
        return (match.points[match.mask, :], match.normals[match.mask, :],
                match.features[:, match.mask], match.mask)


class PointCloudMatcher:
    def __init__(self, target_points, target_normals, target_features, num_neighbors=5,
                 distance_upper_bound=math.inf):
        self.target_points = target_points
        self.target_normals = target_normals
        self.target_features = target_features

        self.num_neighbors = num_neighbors
        self.distance_upper_bound = distance_upper_bound
        self.kdtree_op = KDTreeLayer.setup(target_points)

    @classmethod
    def from_point_cloud(cls, pcl, features, num_neighbors=5,
                         distance_upper_bound=math.inf):
        return cls(pcl.points, pcl.normals, features, num_neighbors=num_neighbors,
                   distance_upper_bound=distance_upper_bound)

    def find_correspondences(self, source_points):

        match_mask = KDTreeLayer.query(source_points, k=self.num_neighbors,
                                       distance_upper_bound=self.distance_upper_bound)
        knn_index = KDTreeLayer.last_query[1]

        top1_knn = knn_index[match_mask, 0]

        matched_points = self.target_points[top1_knn]
        matched_normals = self.target_normals[top1_knn]

        matched_features = self.kdtree_op(self.target_features, source_points)
        matched_features = matched_features[:, match_mask]
        
        return matched_points, matched_normals, matched_features, match_mask
