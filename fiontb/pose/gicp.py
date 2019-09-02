import torch
import torch.optim

from fiontb.spatial.matching import DensePointMatcher
from fiontb.pose.se3 import SE3Exp
from fiontb.camera import Project


class LabICP:
    def __init__(self, num_iters):
        self.num_iters = num_iters

    def estimate(self, target_points, target_mask, target_normals,
                 src_points, kcam, transform):

        exp = SE3Exp.apply

        dtype = target_points.dtype
        device = target_points.device

        transform = torch.tensor([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0]],
                                 dtype=dtype, device=device,
                                 requires_grad=True)

        upsilon_omega = torch.zeros(1, 6, requires_grad=True, device=device)
        optim = torch.optim.SGD([upsilon_omega], 0.0001)
        matcher = DensePointMatcher()

        kcam = kcam.to(device)
        # transform = transform.to(device)

        src_points = src_points.view(-1, 3)
        target_normals = target_normals.view(-1, 3)

        
        for i in range(self.num_iters):
            mpoints, mindex = matcher.match(
                target_points, target_mask,
                src_points, kcam, transform)

            normals = target_normals[mindex]
            import ipdb; ipdb.set_trace()

            diff = mpoints - (src_points @ exp(upsilon_omega).squeeze())[:, :3]
            #diff = mpoints - (src_points @ transform)[:, :3]

            cost = torch.bmm(normals.view(-1, 1, 3), diff.view(-1, 3, 1))

            cost = cost*cost
            cost = cost.sum(0)
            cost.backward()

            optim.zero_grad()
            optim.step()

            # transform = se3_lie_exp(twist)
            print(transform)
            print(cost.item())


def _main():
    from pathlib import Path
    from fiontb.data.ftb import load_ftb
    from fiontb.frame import FramePointCloud

    device = "cuda:0"
    lab = LabICP(2000)

    _TEST_DATA = Path(__file__).parent
    dataset = load_ftb(_TEST_DATA / "_test/sample2")

    frame = dataset[0]
    next_frame = dataset[1]

    fpcl = FramePointCloud.from_frame(frame).to(device)
    next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

    lab.estimate(fpcl.points, fpcl.mask, fpcl.normals, next_fpcl.points,
                 next_fpcl.kcam, torch.eye(4))


if __name__ == '__main__':
    _main()
