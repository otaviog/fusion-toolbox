import math

import torch
from torch.nn.functional import interpolate

from fiontb._cfiontb import (
    icp_estimate_jacobian_gpu, EstimateNormalsMethod, estimate_normals)

from . import se3

import matplotlib.pyplot as plt

# pylint: disable=invalid-name


class _ICPEstimate:
    def __init__(self, scale, num_iters):
        pass

    def estimate(self):
        pass
            
class ICPOdometry:
    def __init__(self, scale_iters):
        self.scale_iters = scale_iters

    def estimate(self, target_points, target_normals, target_mask,
                 source_points, source_mask,
                 kcam, transform):
        device = target_points.device

        kcam = kcam.matrix.to(device)
        transform = transform.to(device)

        target_points = target_points.permute(2, 0, 1)
        target_normals = target_normals.permute(2, 0, 1)

        source_points = source_points.permute(2, 0, 1)

        for scale, num_iters in self.scale_iters:
            if scale < 1.0:
                points0 = interpolate(
                    target_points, scale_factor=(scale, scale), mode='bilinear')
                mask0 = interpolate(target_mask, scale_factor=(scale, scale),
                                    mode='nearest')
                normals = torch.empty(points0.size(1), points0.size(2), 3,
                                      dtype=torch.float,
                                      device=points0.device)
                estimate_normals(
                    points0, mask0, normals,
                    EstimateNormalsMethod.CentralDifferences)

                points1 = interpolate(
                    source_points, scale_factor=(scale, scale))

                mask1 = interpolate(source_mask, scale_factor=(scale, scale))
            else:
                points0 = target_points
                normals0 = target_normals
                mask0 = target_mask

                points1 = source_points
                mask1 = source_mask

            import ipdb; ipdb.set_trace()

            points1 = points1.view(-1, 3)
            mask1 = mask1.view(-1)

            jacobian = torch.zeros(points1.size(0), 6, device=device,
                                   dtype=torch.float)
            residual = torch.zeros(points1.size(0), device=device,
                                   dtype=torch.float)

            for _ in range(num_iters):
                icp_estimate_jacobian_gpu(
                    points0, normals0, mask0,
                    points1, mask1, kcam,
                    transform, jacobian, residual)

                Jt = jacobian.transpose(1, 0)
                JtJ = Jt @ jacobian
                upper_JtJ = torch.cholesky(JtJ.double())

                Jr = Jt @ residual
                update = torch.cholesky_solve(
                    Jr.view(-1, 1).double(), upper_JtJ).squeeze()

                update_matrix = se3.exp(update.cpu()).to(device).float()
                transform = update_matrix @ transform

        return transform


def _show_pcl(pcls):
    import tenviz

    ctx = tenviz.Context(640, 480)

    with ctx.current():
        pcls = [tenviz.create_point_cloud(pcl.points, pcl.colors)
                for pcl in pcls]

    viewer = ctx.viewer(pcls, tenviz.CameraManipulator.WASD)
    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break
        key = chr(key & 0xff)

        if key == '1':
            pcls[0].visible = not pcls[0].visible
        elif key == '2':
            pcls[1].visible = not pcls[1].visible


def _prepare_frame(frame):
    from fiontb.filtering import bilateral_filter_depth_image

    frame.depth_image = bilateral_filter_depth_image(
        frame.depth_image,
        frame.depth_image > 0,
        depth_scale=frame.info.depth_scale)

    return frame


def _test1():
    from pathlib import Path

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud

    torch.set_printoptions(precision=10)

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry([(1.0, 15)])
    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    frame = dataset[0]
    next_frame = dataset[1]

    frame = _prepare_frame(frame)
    next_frame = _prepare_frame(next_frame)

    fpcl = FramePointCloud(frame)
    next_fpcl = FramePointCloud(next_frame)

    relative_rt = icp.estimate(fpcl.points.to(device),
                               fpcl.normals.to(device),
                               fpcl.fg_mask.to(device),
                               next_fpcl.points.to(device),
                               next_fpcl.fg_mask.to(device),
                               next_fpcl.kcam,
                               torch.eye(4).to(device))

    pcl0 = fpcl.unordered_point_cloud(world_space=True)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl1.transform(relative_rt.cpu())
    _show_pcl([pcl0, pcl1])
    return


def _test():
    from pathlib import Path

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud
    from fiontb.viz.datasetviewer import DatasetViewer

    torch.set_printoptions(precision=10)

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry([(1.0, 15)])
    device = "cuda:0"

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)
    prev_frame = _prepare_frame(dataset[0])
    prev_fpcl = FramePointCloud(prev_frame)

    for i in range(1, len(dataset)):
        frame = _prepare_frame(dataset[i])
        fpcl = FramePointCloud(frame)

        relative_rt = icp.estimate(prev_fpcl.points.to(device),
                                   prev_fpcl.normals.to(device),
                                   prev_fpcl.fg_mask.to(device),
                                   fpcl.points.to(device),
                                   fpcl.fg_mask.to(device),
                                   fpcl.kcam,
                                   torch.eye(4).to(device))
        relative_rt = relative_rt.cpu()
        dataset.get_info(
            i).rt_cam = dataset[i-1].info.rt_cam.integrate(relative_rt)

        prev_frame = frame
        prev_fpcl = fpcl

    viewer = DatasetViewer(dataset)
    viewer.run()


if __name__ == '__main__':
    _test()
