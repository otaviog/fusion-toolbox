import torch


def get_rand_se3_mat():
    v0 = torch.rand(3)
    v1 = torch.rand(3)

    v2 = v0.cross(v1)
    v1 = v2.cross(v0)

    v0 /= v0.norm()
    v1 /= v1.norm()
    v2 /= v2.norm()

    se3 = torch.eye(4)
    se3[:3, 0] = v0
    se3[:3, 1] = v1
    se3[:3, 2] = v2

    se3[:3, 3] = torch.rand(3)

    return se3
