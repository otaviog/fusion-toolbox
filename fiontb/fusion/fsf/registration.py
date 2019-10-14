import torch
import scipy.optimize

from fiontb.pose.so3 import SO3tExp
from fiontb.camera import RigidTransform
from fiontb.spatial.kdtree_layer import KDTreeLayer


class _ClosureBox:
    def __init__(self, closure, x, learning_rate=1.0):
        self.closure = closure
        self.loss = None
        self.x = x
        self.learning_rate = learning_rate

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
        self.loss = self.closure()*self.learning_rate
        if self.loss == 0:
            return 0

        self.loss.backward(torch.ones(1, 6, device="cuda:0"))

        grad = self.x.grad.cpu().numpy()

        return grad.transpose().flatten()


class SurfelCloudRegistration:
    def __init__(self, num_iters, learning_rate, use_lbfgs=False):
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.use_lbfgs = use_lbfgs
        self.last_knn_index = None

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

        good_conf = source_surfels.confidences > 0.5
        # source_surfels = source_surfels[good_conf]
        # target_surfels = target_surfels[target_surfels.confidences > 0.5]

        target_features = target_surfels.features
        source_features = source_surfels.features

        target_normals = target_surfels.normals.transpose(1, 0)
        source_normals = source_surfels.normals.view(-1, 3, 1)

        def _closure():
            nonlocal iternum

            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()
            pos = RigidTransform(transform) @ source_surfels.positions

            valid = KDTreeLayer.query(pos, 5)
            self.last_knn_index = KDTreeLayer.last_query[1]

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
            if self.use_lbfgs:
                loss.backward()

            print(iternum, loss.item())
            iternum += 1

            return loss

        if self.use_lbfgs:
            optim.step(_closure)
        else:
            box = _ClosureBox(_closure, upsilon_omega)
            scipy.optimize.fmin_bfgs(box.func, upsilon_omega.detach().cpu().numpy(),
                                     box.grad, maxiter=20,
                                     disp=False, gtol=0.00001)

        return exp(upsilon_omega).detach().squeeze(0)
