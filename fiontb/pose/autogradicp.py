import torch
from tenviz.pose import SE3

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.so3 import SO3tExp
from fiontb.camera import Project, RigidTransform
from fiontb.filtering import FeatureMap


class AutogradICP:
    def __init__(self, num_iters, learning_rate=0.05):
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    def estimate(self, kcam, source_points, target_points=None,
                 target_mask=None, target_normals=None,
                 target_feats=None, source_feats=None,
                 transform=None,
                 geom_weight=0.5, feat_weight=0.5):
        exp = SO3tExp.apply
        proj = Project.apply
        image_map = FeatureMap.apply

        has_geom = not (
            target_points is None or target_normals is None or source_points is None)
        has_feat = not (target_feats is None or source_feats is None)

        source_points = source_points.view(-1, 3)

        if has_geom:
            target_normals = target_normals.view(-1, 3)
            matcher = DensePointMatcher()
            device = target_points.device
            dtype = target_points.dtype
        else:
            geom_weight = 0.0

        if has_feat:
            target_feats = target_feats.view(
                -1, target_feats.size(-2), target_feats.size(-1))
            source_feats = source_feats.squeeze().view(-1, source_points.size(0))
            device = target_feats.device
            dtype = target_feats.dtype
        else:
            feat_weight = 0.0

        kcam = kcam.to(device)
        if transform is None:
            upsilon_omega = torch.zeros(
                1, 6, requires_grad=True, device=device, dtype=dtype)
        else:
            se3 = SE3.from_matrix(transform.cpu())
            upsilon_omega = se3.log(dtype).to(device).to(dtype).view(1, 6)
            upsilon_omega.requires_grad = True

        optim = torch.optim.LBFGS([upsilon_omega], lr=self.learning_rate,
                                  max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)

        total_weight = geom_weight + feat_weight
        geom_weight = geom_weight / total_weight
        feat_weight = feat_weight / total_weight

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()

            geom_loss = 0
            feat_loss = 0
            if has_geom:
                tgt_matched_p3d, matched_index = matcher.match(
                    target_points, target_mask,
                    source_points, kcam, transform)

                valid_matches = matched_index > -1

                tgt_matched_p3d = tgt_matched_p3d[valid_matches]
                src_matched_p3d = source_points[valid_matches]

                matched_index = matched_index[valid_matches]
                tgt_matched_normals = target_normals[matched_index]

                diff = tgt_matched_p3d - \
                    (RigidTransform(transform) @ src_matched_p3d)
                cost = torch.bmm(tgt_matched_normals.view(-1,
                                                          1, 3), diff.view(-1, 3, 1))
                geom_loss = torch.pow(cost, 2).mean()

            if has_feat:
                tgt_uv = proj(RigidTransform(transform)
                              @ source_points, kcam.matrix)
                tgt_feats, bound_mask = image_map(
                    target_feats, tgt_uv)
                bound_mask = bound_mask.detach()

                tgt_feats = tgt_feats[:, bound_mask]
                match_src_feats = source_feats[:, bound_mask]

                res = torch.norm(tgt_feats - match_src_feats, 2, dim=0)
                feat_loss = res.mean()

            loss = geom_loss*geom_weight + feat_loss*feat_weight

            loss.backward()
            print(loss)
            return loss

        optim.step(closure)

        return exp(upsilon_omega).detach().squeeze(0)
