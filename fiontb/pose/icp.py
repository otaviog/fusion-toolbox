import math

import torch

from fiontb._cfiontb import icp_estimate_jacobian_gpu
from . import se3

import matplotlib.pyplot as plt

# pylint: disable=invalid-name


class ICPOdometry:
    def __init__(self, scale_iters):
        self.scale_iters = scale_iters

    def estimate(self, points0, normals0, mask0,
                 points1, mask1,
                 kcam, transform):

        plt.figure()
        plt.imshow(points0.cpu().numpy())
        plt.figure()
        plt.imshow(points1.cpu().numpy())
        # plt.show()

        device = points0.device

        points1 = points1.view(-1, 3)
        jacobian = torch.zeros(points1.size(0), 6, device=device,
                               dtype=torch.float)
        residual = torch.zeros(points1.size(0), device=device,
                               dtype=torch.float)
        mask1 = mask1.view(-1)
        kcam = kcam.matrix.to(device)
        transform = transform.to(device)
        for scale, num_iters in self.scale_iters:
            # TODO: Scale image
            for _ in range(num_iters):
                icp_estimate_jacobian_gpu(
                    points0, normals0, mask0,
                    points1, mask1, kcam,
                    transform, jacobian, residual)

                J = jacobian.double()
                r = residual.double()
                print(_, r.mean())
                Jt = J.transpose(1, 0)
                JtJ = Jt @ J
                inv_JtJ = JtJ.cpu().inverse()
                Jr = Jt @ r

                update = inv_JtJ @ Jr.cpu()
                update_matrix = se3.exp(update).to(device).float()
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
