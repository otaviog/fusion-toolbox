import math
import torch
import scipy.optimize

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.so3 import SO3tExp
from fiontb.camera import Project, RigidTransform
from fiontb.filtering import FeatureMap
from fiontb.downsample import (DownsampleXYZMethod)

from .multiscale_optim import MultiscaleOptimization as _MultiscaleOptimization


class _ClosureBox:
    def __init__(self, closure, x):
        self.closure = closure
        self.loss = None
        self.x = x

    def func(self, x):
        with torch.no_grad():
            self.x[0][0] = float(x[0])
            self.x[0][1] = float(x[1])
            self.x[0][2] = float(x[2])
            self.x[0][3] = float(x[3])
            self.x[0][4] = float(x[4])
            self.x[0][5] = float(x[5])
        self.loss = self.closure()
        if self.loss == 0:
            return 0

        return self.loss.detach().cpu().item()

    def grad(self, x):
        with torch.no_grad():
            self.x[0][0] = float(x[0])
            self.x[0][1] = float(x[1])
            self.x[0][2] = float(x[2])
            self.x[0][3] = float(x[3])
            self.x[0][4] = float(x[4])
            self.x[0][5] = float(x[5])
        self.loss = self.closure()*0.05
        if self.loss == 0:
            return 0

        self.loss.backward(torch.ones(1, 6, device="cuda:0"))

        grad = self.x.grad.cpu().numpy()

        return grad.transpose().flatten()


def _to_4x4(mtx):
    if mtx.size(0) == 4 and mtx.size(1) == 4:
        return mtx

    ret = torch.eye(4, device=mtx.device, dtype=mtx.dtype)

    ret[:3, :4] = mtx[:3, :4]

    return ret


class AutogradICP:
    def __init__(self, num_iters, learning_rate=0.05, use_lbfgs=False):
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.use_lbfgs = use_lbfgs

    def estimate(self, kcam, source_points, source_mask,
                 target_points=None,
                 target_mask=None, target_normals=None,
                 target_feats=None, source_feats=None,
                 transform=None, geom_weight=0.5, feat_weight=0.5):
        has_geom = not (
            target_points is None or target_normals is None or source_points is None)
        has_feat = not (target_feats is None or source_feats is None)

        source_points = source_points[source_mask].view(-1, 3)

        matcher = DensePointMatcher()
        if has_geom:
            target_normals = target_normals.view(-1, 3)
            device = target_points.device
            dtype = target_points.dtype
        else:
            geom_weight = 0.0

        if has_feat:
            target_feats = target_feats.view(
                -1, target_feats.size(-2), target_feats.size(-1))
            source_feats = (source_feats[:, source_mask].
                            view(-1, source_points.size(0)))
            device = target_feats.device
            dtype = target_feats.dtype
        else:
            feat_weight = 0.0

        kcam = kcam.to(device)
        upsilon_omega = torch.zeros(
            1, 6, requires_grad=True, device=device, dtype=dtype)
        initial_transform = None
        if transform is not None:
            # TODO: investigate non-orthogonal matrices erro
            # se3 = SE3.from_matrix(transform.cpu())
            # upsilon_omega = se3.log().to(device).to(dtype).view(1, 6)
            # upsilon_omega.requires_grad = True
            source_points = RigidTransform(transform) @ source_points
            initial_transform = transform

        optim = torch.optim.LBFGS([upsilon_omega], lr=self.learning_rate,
                                  max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)

        total_weight = geom_weight + feat_weight
        geom_weight = geom_weight / total_weight
        feat_weight = feat_weight / total_weight

        best_loss = math.inf
        best_params = upsilon_omega

        def _closure():
            nonlocal best_loss
            nonlocal best_params

            optim.zero_grad()
            transform = SO3tExp.apply(upsilon_omega).squeeze()

            geom_loss = 0
            feat_loss = 0

            tgt_matched_p3d, matched_index = matcher.match(
                target_points, target_mask,
                source_points, kcam, transform)

            valid_matches = matched_index > -1
            src_matched_p3d = source_points[valid_matches]
            trans_src = (RigidTransform(transform) @ src_matched_p3d)

            if has_geom:
                tgt_matched_p3d = tgt_matched_p3d[valid_matches]

                matched_index = matched_index[valid_matches]
                tgt_matched_normals = target_normals[matched_index]

                diff = tgt_matched_p3d - trans_src

                cost = torch.bmm(tgt_matched_normals.view(-1,
                                                          1, 3), diff.view(-1, 3, 1))
                geom_loss = torch.pow(cost, 2).mean()

            if has_feat:
                tgt_uv = Project.apply(trans_src, kcam.matrix)
                # tgt_uv2 = Project.apply(RigidTransform(transform) @ source_points,
                #                        kcam.matrix)
                # _, bmask = FeatureMap.apply(
                #    target_feats, tgt_uv2)

                tgt_feats, bound_mask = FeatureMap.apply(
                    target_feats, tgt_uv)
                bound_mask = bound_mask.detach()

                tgt_feats = tgt_feats[:, bound_mask]
                match_src_feats = source_feats[:, valid_matches][:, bound_mask]

                res = torch.norm(tgt_feats - match_src_feats, 2, dim=0)
                feat_loss = res.mean()

            loss = geom_loss*geom_weight + feat_loss*feat_weight

            if torch.isnan(loss):
                return 0

            if self.use_lbfgs:
                loss.backward()

            if loss < best_loss:
                best_loss = loss.clone()
                best_params = upsilon_omega.detach().clone().squeeze(0)

            return loss

        if self.use_lbfgs:
            optim.step(_closure)
        else:
            box = _ClosureBox(_closure, upsilon_omega)
            scipy.optimize.fmin_bfgs(box.func, upsilon_omega.detach().cpu().numpy(),
                                     box.grad, #maxiter=self.num_iters
                                     disp=False)
        transform = SO3tExp.apply(best_params).squeeze(0)
        transform = _to_4x4(transform)

        if initial_transform is not None:
            transform = transform @ _to_4x4(initial_transform)

        return transform, True


class MultiscaleAutogradICP(_MultiscaleOptimization):
    def __init__(self, scale_iters_lr, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        super().__init__(
            [(scale, AutogradICP(iters, lr), use_feats)
             for scale, iters, lr, use_feats in scale_iters_lr],
            downsample_xyz_method)
