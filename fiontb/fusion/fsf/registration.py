from scipy.spatial.ckdtree import cKDTree
import torch

from fiontb.pose.so3 import SO3tExp
from fiontb.camera import RigidTransform
from fiontb.spatial.kdtree_layer import KDTreeLayer


class SurfelCloudRegistration:
    def __init__(self, num_iters, learning_rate):
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    def estimate(self, target_surfels, source_surfels):
        dtype = target_surfels.positions.dtype
        device = target_surfels.positions.device

        map_op = KDTreeLayer.setup(target_surfels.positions)

        upsilon_omega = torch.zeros(
            1, 6, requires_grad=True, device=device, dtype=dtype)
        optim = torch.optim.LBFGS([upsilon_omega], lr=self.learning_rate,
                                  max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)
        exp = SO3tExp.apply

        iternum = 0

        good_conf = source_surfels.confidences > 0.75

        source_surfels = source_surfels[good_conf]

        target_features = target_surfels.colors.transpose(1, 0).float()/255
        source_features = source_surfels.colors.transpose(1, 0).float()/255

        target_normals = target_surfels.normals.transpose(1, 0)
        source_normals = source_surfels.normals.view(-1, 3, 1)

        def _closure():
            nonlocal iternum

            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()
            pos = RigidTransform(transform) @ source_surfels.positions

            valid = KDTreeLayer.query(pos, 5, 1)
            tgt_norm = map_op(target_normals, pos)
            tgt_norm = tgt_norm[:, valid]
            tgt_norm = tgt_norm.transpose(1, 0).view(-1, 1, 3)

            norm_loss = 1.0 - tgt_norm.bmm(source_normals[valid, :]).squeeze()
            norm_loss = torch.pow(norm_loss, 2).mean()

            tgt_feat = map_op(target_features, pos)
            tgt_feat = tgt_feat[:, valid]
            diff = tgt_feat - source_features[:, valid]
            feat_loss = diff.norm(dim=0).mean()

            loss = feat_loss + norm_loss
            loss.backward()

            if torch.isnan(loss):
                import ipdb
                ipdb.set_trace()

            print(iternum, loss.item())
            iternum += 1
            return loss

        optim.step(_closure)
        return exp(upsilon_omega).detach().squeeze(0)
