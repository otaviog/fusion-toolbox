from scipy.spatial.ckdtree import cKDTree
from pykdtree.kdtree import KDTree
import torch.autograd
import numpy


from fiontb._cfiontb import FeatureMap3DOp


class KDTreeLayer(torch.autograd.Function):
    tree = None
    target_xyz = None
    last_query = (None, None)

    _gradcheck = False

    @staticmethod
    def setup(target_xyz):
        KDTreeLayer.target_xyz = target_xyz
        KDTreeLayer.tree = cKDTree(
            target_xyz.cpu().numpy(),
            balanced_tree=True
        )
        return KDTreeLayer.apply

    @staticmethod
    def query(xyz, k=1, distance_upper_bound=numpy.inf):
        ref_device = xyz.device
        ref_dtype = xyz.dtype

        distance, index = KDTreeLayer.tree.query(
            xyz.detach().cpu().numpy(), k=k,
            distance_upper_bound=distance_upper_bound)

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
                                   dtype=ref_dtype,
                                   device=ref_device)

        FeatureMap3DOp.forward(distances, index, features, out_features)
        return out_features

    def backward(ctx, dl_feat):
        source_xyz, index, features = ctx.saved_tensors

        ref_dtype = features.dtype
        ref_device = features.device

        epsilon_distances = torch.empty(index.size(0), 6, index.size(1),
                                        dtype=ref_dtype,
                                        device=ref_device)
        FeatureMap3DOp.compute_epsilon_distances(
            KDTreeLayer.target_xyz, source_xyz,
            index, epsilon_distances)

        dl_xyz = torch.empty(index.size(0), 3,
                             dtype=ref_dtype,
                             device=ref_device)

        FeatureMap3DOp.backward(epsilon_distances, index,
                                features, dl_feat, dl_xyz)

        return None, dl_xyz
