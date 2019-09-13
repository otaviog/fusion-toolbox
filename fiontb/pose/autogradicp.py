import torch

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.so3 import SO3tExp
from fiontb.camera import Project, Homogeneous
from fiontb.filtering import ImageMap, FeatureMap


class AutogradICP:
    def __init__(self, num_iters, learning_rate=0.05):
        self.num_iters = num_iters
        self.learning_rate = learning_rate

    def estimate(self, kcam, src_points, tgt_image_p3d=None, tgt_mask=None, tgt_normals=None,
                 tgt_image_feat=None, src_image_feat=None,
                 geom_weight=0.5, feat_weight=0.5):
        torch.backends.cudnn.enabled = False

        exp = SO3tExp.apply
        proj = Project.apply
        # image_map = ImageMap.apply
        image_map = FeatureMap.apply

        has_geom = not (
            tgt_image_p3d is None or tgt_normals is None or src_points is None)
        has_feat = not (tgt_image_feat is None or src_image_feat is None)

        src_points = src_points.view(-1, 3)

        if has_geom:
            tgt_normals = tgt_normals.view(-1, 3)
            matcher = DensePointMatcher()
            device = tgt_image_p3d.device
            dtype = tgt_image_p3d.dtype
        else:
            geom_weight = 0.0

        if has_feat:
            tgt_image_feat = tgt_image_feat.view(
                -1, tgt_image_feat.size(-2), tgt_image_feat.size(-1))
            src_image_feat = src_image_feat.squeeze().view(-1, src_points.size(0))
            device = tgt_image_feat.device
            dtype = tgt_image_feat.dtype
        else:
            feat_weight = 0.0

        kcam = kcam.to(device)
        upsilon_omega = torch.zeros(
            1, 6, requires_grad=True, device=device, dtype=dtype)
        optim = torch.optim.LBFGS([upsilon_omega], lr=self.learning_rate, max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)

        total_weight = geom_weight + feat_weight
        geom_weight = geom_weight / total_weight
        feat_weight = feat_weight / total_weight

        recompute_grad = True

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()
            nonlocal recompute_grad

            geom_loss = 0
            feat_loss = 0
            if has_geom:
                tgt_matched_p3d, matched_index = matcher.match(
                    tgt_image_p3d, tgt_mask,
                    src_points, kcam, transform)

                valid_matches = matched_index > -1

                tgt_matched_p3d = tgt_matched_p3d[valid_matches]
                src_matched_p3d = src_points[valid_matches]

                matched_index = matched_index[valid_matches]
                tgt_matched_normals = tgt_normals[matched_index]

                diff = tgt_matched_p3d - \
                    (Homogeneous(transform) @ src_matched_p3d)
                cost = torch.bmm(tgt_matched_normals.view(-1,
                                                          1, 3), diff.view(-1, 3, 1))
                geom_loss = torch.pow(cost, 2).mean()

            if has_feat:
                tgt_uv = proj(Homogeneous(transform) @ src_points, kcam.matrix)
                tgt_feats, bound_mask = image_map(
                    tgt_image_feat, tgt_uv)
                bound_mask = bound_mask.detach()
                recompute_grad = False

                tgt_feats = tgt_feats[:, bound_mask]
                match_src_feats = src_image_feat[:, bound_mask]

                res = torch.norm(tgt_feats - match_src_feats, 2, dim=0)
                feat_loss = res.mean()

            loss = geom_loss*geom_weight + feat_loss*feat_weight

            loss.backward()
            print(loss)

            return loss

        optim.step(closure)

        return exp(upsilon_omega).detach().squeeze(0)
