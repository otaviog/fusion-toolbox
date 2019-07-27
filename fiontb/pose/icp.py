"""Pose estimation via iterative closest points algorithm.
"""

import torch

from tenviz.pose import SE3

from fiontb.downsample import (
    downsample_xyz, downsample, DownsampleXYZMethod, DownsampleMethod)
from fiontb._cfiontb import icp_estimate_jacobian_gpu
from fiontb._utils import empty_ensured_size

# pylint: disable=invalid-name


class ICPOdometry:
    """Point-to-plane iterative closest points
    algorithm.

    Attributes:

        num_iters (int): Number of iterations for the Gauss-Newton
         optimization algorithm.

    """

    def __init__(self, num_iters):
        self.num_iters = num_iters

        self.jacobian = None
        self.residual = None

    def estimate(self, target_points, target_normals, target_mask,
                 source_points, source_mask, kcam, transform=None):
        """Estimate the ICP odometry between a target points and normals in a
        grid against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d **normals** that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A uint8 [WxH] mask
              tensor of valid target points.

            source_points (:obj:`torch.Tensor`): A float [Nx3] tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): A uint8 [N] mask
             tensor of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A float [4x4] initial
             transformation matrix.

        Returns: (:obj:`torch.Tensor`): A [4x4] rigid motion matrix
            that aligns source points to target points.

        """

        device = source_points.device

        kcam = kcam.matrix.to(device)
        if transform is None:
            transform = torch.eye(4, device=device)
        else:
            transform = transform.to(device)

        source_points = source_points.view(-1, 3)
        source_mask = source_mask.view(-1)

        self.jacobian = empty_ensured_size(self.jacobian, source_points.size(0), 6,
                                           device=device, dtype=torch.float)
        self.residual = empty_ensured_size(self.residual, source_points.size(0),
                                           device=device, dtype=torch.float)

        for _ in range(self.num_iters):
            icp_estimate_jacobian_gpu(target_points, target_normals, target_mask,
                                      source_points, source_mask, kcam, transform,
                                      self.jacobian, self.residual)

            Jt = self.jacobian.transpose(1, 0)
            JtJ = Jt @ self.jacobian
            upper_JtJ = torch.cholesky(JtJ.double())

            Jr = Jt @ self.residual

            update = torch.cholesky_solve(
                Jr.view(-1, 1).double(), upper_JtJ).squeeze()

            update_matrix = SE3.exp(
                update.cpu()).to_matrix().to(device).float()
            transform = update_matrix @ transform

        return transform


class MultiscaleICPOdometry:
    """Pyramidal point-to-plane iterative closest points
    algorithm.
    """

    def __init__(self, scale_iters, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Initialize the multiscale ICP.

        Args:

            scale_iters (List[(float, int)]): Scale levels and its
             number of iterations. Scales must be <= 1.0

            downsample_xyz_method
             (:obj:`fiontb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the xyz target and source points.
        """

        self.scale_icp = [(scale, ICPOdometry(iters))
                          for scale, iters in scale_iters]
        self.downsample_xyz_method = downsample_xyz_method

    def estimate(self, target_points, target_normals, target_mask,
                 source_points, source_mask,
                 kcam, transform=None):
        """Estimate the ICP odometry between a target frame points and normals
        against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d normals that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A uint8 [WxH] mask tensor
             of valid target points.

            source_points (:obj:`torch.Tensor`): A float [WxHx3] tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): Uint8 [WxH] mask tensor
             of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A float [4x4] initial
             transformation matrix.

        Returns: (:obj:`torch.Tensor`): A [4x4] rigid motion matrix
            that aligns source points to target points.

        """

        if transform is None:
            transform = torch.eye(4, device=target_points.device)

        for scale, icp_instance in self.scale_icp:
            if scale < 1.0:
                tgt_points = downsample_xyz(target_points, target_mask, scale,
                                            method=self.downsample_xyz_method)
                tgt_normals = downsample_xyz(target_normals, target_mask, scale,
                                             normalize=True,
                                             method=self.downsample_xyz_method)
                tgt_mask = downsample(target_mask, scale,
                                      DownsampleMethod.Nearest)

                src_points = downsample_xyz(source_points, source_mask, scale,
                                            normalize=False,
                                            method=self.downsample_xyz_method)
                src_mask = downsample(
                    source_mask, scale, DownsampleMethod.Nearest)
            else:
                tgt_points = target_points
                tgt_normals = target_normals
                tgt_mask = target_mask

                src_points = source_points
                src_mask = source_mask

            scaled_kcam = kcam.scaled(scale)
            transform = icp_instance.estimate(tgt_points, tgt_normals, tgt_mask,
                                              src_points, src_mask, scaled_kcam,
                                              transform)

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


def _test_geom1():
    from pathlib import Path

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry(15)
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
                               next_fpcl.kcam)

    pcl0 = fpcl.unordered_point_cloud(world_space=True)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl1.transform(relative_rt.cpu())
    _show_pcl([pcl0, pcl1])


def _test_geom2():
    from pathlib import Path

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud
    from fiontb.viz.datasetviewer import DatasetViewer

    torch.set_printoptions(precision=10)

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = ICPOdometry(15)
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
                                   fpcl.kcam)
        relative_rt = relative_rt.cpu()
        dataset.get_info(
            i).rt_cam = dataset[i-1].info.rt_cam.integrate(relative_rt)

        prev_frame = frame
        prev_fpcl = fpcl

    viewer = DatasetViewer(dataset)
    viewer.run()


def _test_multiscale_geom():
    from pathlib import Path

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    icp = MultiscaleICPOdometry([(0.25, 15), (0.5, 10), (1.0, 5)])
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
                               next_fpcl.kcam)

    pcl0 = fpcl.unordered_point_cloud(world_space=True)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl1.transform(relative_rt.cpu())
    _show_pcl([pcl0, pcl1])


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=[
        'geometric1', 'geometric2',
        'multiscale-geometric'])

    args = parser.parse_args()

    if args.test == 'geometric1':
        _test_geom1()
    elif args.test == 'geometric2':
        _test_geom2()
    elif args.test == 'multiscale-geometric':
        _test_multiscale_geom()


if __name__ == '__main__':
    _main()
