import torch

from fiontb._cfiontb import icp_estimate_jacobian_gpu


class ICPOdometry:
    def __init__(self, scale_iters):
        self.scale_iters = scale_iters

    def estimate(self, points0, normals0,
                 points1, kcam, init_mtx):
        import ipdb
        ipdb.set_trace()

        jacobian = torch.zeros(points1.size(0), 7, device=points0.device, dtype=torch.float)
        residual = torch.zeros(points1.size(0), device=points0.device, dtype=torch.float)
        params = torch.zeros(7, device=points0.device, dtype=torch.float)
        
        for scale, num_iters in self.scale_iters:
            # TODO: Scale image
            for _ in range(num_iters):
                icp_estimate_jacobian_gpu(
                    points0, normals0, points1.view(-1, 3),
                    kcam, params, jacobian, residual)
                JtJ = jacobian.transpose(1, 0).matmul(jacobian)
                inv_JtJ = JtJ.inverse()
                Jr = jacobian @ residual

                update = inv_JtJ @ Jr

                prev_residual, curr_residual = curr_residual, prev_residual


def _test():
    from pathlib import Path
    import torch

    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud
    from fiontb.viz.datasetviewer import DatasetViewer

    _TEST_DATA = Path(__file__).parent / "_test"
    dataset = load_ftb(_TEST_DATA / "sample1")

    dataset.get_info(0).rt_cam.matrix = torch.eye(4)

    icp = ICPOdometry([(1.0, 15)])
    device = "cuda:0"
    for i in range(1, len(dataset)):
        frame = FramePointCloud(dataset[i])
        prev_frame = FramePointCloud(dataset[i-1])

        relative_rt = icp.estimate(frame.points.to(device),
                                   frame.normals.to(device),
                                   prev_frame.points.to(device),
                                   prev_frame.kcam,
                                   torch.eye(4).to(device))
        dataset.get_info(
            i).rt_cam = prev_frame.info.rt_cam.integrate(relative_rt)

    viewer = DatasetViewer(dataset)
    viewer.run()


if __name__ == '__main__':
    _test()
