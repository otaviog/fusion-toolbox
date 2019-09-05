import torch
import torch.optim

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.so3 import SO3tExp
from fiontb.camera import Project, Homogeneous


def _valid_projs(uv, width, height):
    return ((uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))


class Image(torch.nn.Module):
    def forward(self, image, uv):
        valid_uvs = _valid_projs(uv, image.size(2), image.size(3))
        uv = uv[valid_uvs, :]
        uv = uv.long()

        return image[:, :, uv[:, 1], uv[:, 0]].squeeze(), valid_uvs


class Image2(torch.nn.Module):
    def forward(self, image, uv):
        valid_uvs = _valid_projs(uv, image.size(1), image.size(0))
        uv = uv[valid_uvs, :]
        uv = uv.long()

        return image[uv[:, 1], uv[:, 0]].squeeze(), valid_uvs


class Image3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, uv):
        valid_uvs = _valid_projs(uv, image.size(1), image.size(0))
        uv = uv[valid_uvs, :]
        uv = uv.long()

        ctx.save_for_backward(image, uv, valid_uvs)
        return image[uv[:, 1], uv[:, 0]].squeeze(), valid_uvs

    @staticmethod
    def backward(ctx, dy_image, dy_uv):

        image, uv, valid_uvs = ctx.saved_tensors
        dtype = dy_image.dtype
        device = dy_image.device
        convx = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        convx.weight = torch.nn.Parameter(torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3))

        convy = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        convy.weight = torch.nn.Parameter(torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3))

        xgrad = convx(image.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
        ygrad = convy(image.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()

        uv2 = torch.zeros(valid_uvs.size(0), 2, device=device)
        # import ipdb; ipdb.set_trace()

        xgrad = xgrad[uv[:, 1], uv[:, 0]]
        ygrad = ygrad[uv[:, 1], uv[:, 0]]
        uv2[valid_uvs, 0] = xgrad
        uv2[valid_uvs, 1] = ygrad

        return None, uv2.to("cuda:0")


class LabICP:
    def __init__(self, num_iters):
        self.num_iters = num_iters

    def estimate_geo(self, target_points, target_mask, target_normals,
                     src_points, kcam, transform):
        exp = SO3tExp.apply

        dtype = target_points.dtype
        device = target_points.device

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        optim = torch.optim.LBFGS([upsilon_omega], lr=0.05)
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

        for i in range(self.num_iters):
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
        optim = torch.optim.LBFGS([upsilon_omega], lr=0.05)

        kcam = kcam.to(device)
        src_points = src_points.view(-1, 3)
        src_image = src_image.squeeze().view(-1)

        image = Image3.apply

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()

            tgt_uv = proj(Homogeneous(transform) @ src_points, kcam.matrix)

            tgt_feats, selected = image(target_image, tgt_uv)

            match_src_feats = src_image[selected]

            res = torch.pow(tgt_feats - match_src_feats, 2)
            loss = res.sum()

            loss.backward(retain_graph=True)
            print(loss)

            return loss

        for i in range(self.num_iters):
            optim.step(closure)
            print(upsilon_omega)

        return exp(upsilon_omega).detach().squeeze(0)

    def estimate_feature(self, target_img_points, target_img_feats, target_mask, target_normals,
                         src_points, src_feats, kcam):
        exp = SO3tExp.apply
        proj = Project.apply

        device = target_img_points.device

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        optim = torch.optim.LBFGS([upsilon_omega], lr=0.05)

        kcam = kcam.to(device)
        src_points = src_points.view(-1, 3)
        src_feats = src_feats.squeeze().view(64, -1)

        image = Image()

        def closure():
            optim.zero_grad()
            transform = exp(upsilon_omega).squeeze()

            tgt_uv = proj(Homogeneous(transform) @ src_points, kcam.matrix)

            tgt_feats, selected = image(target_img_feats, tgt_uv)

            match_src_feats = src_feats[:, selected]

            res = torch.norm(tgt_feats - match_src_feats, 2, dim=0)
            loss = res.sum()

            loss.backward(retain_graph=True)
            print(loss)
            import ipdb
            ipdb.set_trace()

            return loss

        for i in range(self.num_iters):
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
    lab = LabICP(5)

    _TEST_DATA = Path(__file__).parent
    dataset = load_ftb(_TEST_DATA / "_test/sample1")

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
