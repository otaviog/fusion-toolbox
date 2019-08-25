import torch
import torch.optim

# from fiontb._cfiontb import dense_point_matcher


def se3_lie_exp(twist):
    rot = twist[:3]

    import ipdb; ipdb.set_trace()
    omega = torch.tensor([[0, -rot[2], rot[1]],
                          [rot[2], 0, -rot[0]],
                          [-rot[1], rot[0], 0.0]])
    omega_sq = omega @ omega
    theta_sq = torch.dot(rot, rot)
    theta = torch.sqrt(theta_sq)

    so3 = torch.eye(3)
    so3 += (torch.sin(theta) / theta) * omega
    so3 += ((1 - torch.cos(theta)) / theta_sq) * omega_sq

    return so3


class LabICP:
    def __init__(self, num_iters):
        self.num_iters = num_iters

    def estimate(self, target_points, target_normals,
                 src_points, kcam, transform):

        optim = torch.optim.SGD(twist, 0.01)

        for i in range(self.num_iters):
            match = dense_point_matcher(
                target_points, src_points, kcam, transform)
            cost = torch.dot(
                target_normals, (match - src_points @ se3_lie_exp(twist)))
            optim.zero_grad()
            cost.backward()
            optim.step()

            transform = se3_lie_exp(twist)


def _main():
    twist = torch.rand(3, requires_grad=True)
    grad = se3_lie_exp(twist)
    import ipdb
    ipdb.set_trace()

    print(grad)

if __name__ == '__main__':
    _main()
