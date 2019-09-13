import torch
import torch.optim

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.so3 import SO3tExp
from fiontb.camera import Project, Homogeneous
from fiontb.filtering import ImageGradient, MMImageGradient


class LabICP:
    def __init__(self, num_iters):
        self.num_iters = num_iters

    def estimate_geo(self, target_points, target_mask, target_normals,
                     src_points, kcam, transform):
        exp = SO3tExp.apply

        dtype = target_points.dtype
        device = target_points.device

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        optim = torch.optim.LBFGS(
            [upsilon_omega], lr=0.005, max_iter=self.num_iters)
        matcher = DensePointMatcher()

        kcam = kcam.to(device)
        src_points = src_points.view(-1, 3)
        target_normals = target_normals.view(-1, 3)

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()
            mpoints, mindex = matcher.match(
                target_points, target_mask,
                src_points, kcam, transform)
            normals = target_normals[mindex]

            diff = mpoints - (Homogeneous(transform) @ src_points)
            cost = torch.bmm(normals.view(-1, 1, 3), diff.view(-1, 3, 1))
            loss = torch.pow(cost, 2).sum()
            print(loss.item())
            loss.backward()

            return loss

        optim.step(closure)
        print(upsilon_omega)

        return exp(upsilon_omega).detach().squeeze(0)

    def estimate_intensity(
            self, target_img_points, target_image, target_mask, target_normals,
            src_points, src_image, kcam):
        exp = SO3tExp.apply
        proj = Project.apply

        device = target_img_points.device

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        #upsilon_omega = torch.rand(1, 6, requires_grad=True, device=device)
        # upsilon_omega = torch.tensor([[0.0017, -0.0015, -0.0009, 0.0027, -0.0029, -0.0004]],
        # requires_grad=True, device=device)
        # upsilon_omega = torch.tensor([[0.0015, -0.0015, -0.0009, 0.0027, -0.0019, -0.0004]],
        # requires_grad=True, device=device)
        optim = torch.optim.LBFGS([upsilon_omega], lr=0.01, max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)

        kcam = kcam.to(device)
        src_points = src_points.view(-1, 3)
        src_image = src_image.squeeze().view(-1)
        target_image = target_image.view(-1,
                                         target_image.size(-2), target_image.size(-1))
        image = MMImageGradient.apply

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()

            tgt_uv = proj(Homogeneous(transform) @ src_points, kcam.matrix)

            tgt_feats, selected = image(target_image, tgt_uv)

            match_src_feats = src_image[selected]

            res = torch.pow(tgt_feats - match_src_feats, 2)
            loss = res.mean()

            loss.backward()
            print(loss)

            return loss

        optim.step(closure)

        return exp(upsilon_omega).detach().squeeze(0)

    def estimate_feature(self, target_img_points, target_img_feats, target_mask, target_normals,
                         src_points, src_feats, kcam):
        torch.backends.cudnn.enabled = False
        #######################################
        # Feature
        #############
        exp = SO3tExp.apply
        proj = Project.apply

        device = target_img_points.device

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        # upsilon_omega = torch.tensor([[ 0.0981,  0.1778, -0.0074, -0.7511,  0.2179, -0.2110]],
        #                             requires_grad=True, device=device)
        #upsilon_omega = torch.tensor(
        #    [[-6.7892e-04, -1.3931e-03,  6.4001e-05,  2.7329e-03, -2.2463e-03,
        #      -4.1050e-04]],
        #    requires_grad=True, device=device)
        optim = torch.optim.LBFGS([upsilon_omega], lr=0.01, max_iter=self.num_iters,
                                  history_size=500, max_eval=4000)

        kcam = kcam.to(device)
        src_points = src_points.view(-1, 3)
        src_feats = src_feats.squeeze().view(-1, src_points.size(0))

        image = MMImageGradient.apply

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()
            tgt_uv = proj(Homogeneous(transform) @ src_points, kcam.matrix)

            tgt_feats, selected = image(target_img_feats, tgt_uv)
            match_src_feats = src_feats[:, selected]

            import ipdb; ipdb.set_trace()
            res = torch.norm(tgt_feats - match_src_feats, 2, dim=0)

            #res = (tgt_feats - match_src_feats).sum(dim=0)
            #import ipdb; ipdb.set_trace()
            #res = torch.norm(tgt_uv, 2, dim=0)
            #res = torch.norm(target_img_points.view(-1, 3) - Homogeneous(transform) @ src_points, 2, dim=0)
            loss = res.mean()

            loss.backward()
            print(loss)

            return loss

        optim.step(closure)
        print(upsilon_omega)

        return exp(upsilon_omega).detach().squeeze(0)


def _prepare_frame(frame, bi_filter=True):
    from fiontb.filtering import bilateral_filter_depth_image
    if bi_filter:
        frame.depth_image = bilateral_filter_depth_image(
            frame.depth_image,
            frame.depth_image > 0,
            depth_scale=frame.info.depth_scale)

    return frame


def _main():
    from pathlib import Path
    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud
    from fiontb.viz.show import show_pcls

    device = "cuda:0"
    lab = LabICP(500)

    _TEST_DATA = Path(__file__).parent
    dataset = load_ftb(_TEST_DATA / "_test/sample2")

    frame = _prepare_frame(dataset[0])
    next_frame = _prepare_frame(dataset[1])

    fpcl = FramePointCloud.from_frame(frame).to(device)
    next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

    relative_rt = lab.estimate_geo(fpcl.points, fpcl.mask, fpcl.normals, next_fpcl.points,
                                   next_fpcl.kcam, torch.eye(4))

    pcl0 = fpcl.unordered_point_cloud(world_space=False)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl2 = pcl1.clone()

    pcl2 = pcl2.transform(relative_rt)
    show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    _main()
