"""Diffentiable layer for KDTree search.
"""

from pykdtree.kdtree import KDTree
import torch.autograd
import numpy

from fiontb._cfiontb import (
    NearestNeighborsOp as _NearestNeighborsOp)


class KDTreeLayer(torch.autograd.Function):
    """Layer for retrieving features associated with 3D-points. 

    The usage is:

    * First call `setup()` with the target points and its features.

    * Use `query()` to query points

    * Use `forward()` to retrieve features interpolated by xyz points.

    """
    tree = None
    target_xyz = None
    last_query = (None, None)

    _gradcheck = False

    @staticmethod
    def setup(target_xyz):
        KDTreeLayer.target_xyz = target_xyz
        KDTreeLayer.tree = KDTree(
            target_xyz.cpu().numpy(),
            leafsize=32)
        return KDTreeLayer.apply

    @staticmethod
    def query(xyz, k=1, distance_upper_bound=numpy.inf):
        ref_device = xyz.device
        ref_dtype = xyz.dtype

        distance, index = KDTreeLayer.tree.query(
            xyz.detach().cpu().numpy(), k=k,
            distance_upper_bound=distance_upper_bound,
            eps=0.5, sqr_dists=True)

        distance = torch.from_numpy(distance).to(
            ref_device).to(ref_dtype).view(-1, k)
        index = torch.from_numpy(index.astype(
            numpy.int64)).to(ref_device).view(-1, k)

        valid = index[:, 0] < KDTreeLayer.tree.n

        KDTreeLayer.last_query = (distance, index)

        return valid

    @staticmethod
    def forward(ctx, features, xyz):
        if KDTreeLayer._gradcheck:
            KDTreeLayer.query(xyz, k=8)

        distances, index = KDTreeLayer.last_query

        ref_dtype = features.dtype
        ref_device = features.device

        ctx.save_for_backward(xyz, index, features)
        out_features = torch.empty(features.size(0), xyz.size(0),
                                   dtype=ref_dtype, device=ref_device)

        _NearestNeighborsOp.forward(distances, index, features, out_features)
        return out_features

    @staticmethod
    def backward(ctx, dl_feat):
        source_xyz, index, features = ctx.saved_tensors

        ref_dtype = features.dtype
        ref_device = features.device

        epsilon_distances = torch.empty(index.size(0), 6, index.size(1),
                                        dtype=ref_dtype,
                                        device=ref_device)
        _NearestNeighborsOp.compute_epsilon_distances(
            KDTreeLayer.target_xyz, source_xyz,
            index, epsilon_distances)

        dl_xyz = torch.empty(index.size(0), 3,
                             dtype=ref_dtype,
                             device=ref_device)

        _NearestNeighborsOp.backward(epsilon_distances, index,
                                     features, dl_feat, dl_xyz)

        return None, dl_xyz
