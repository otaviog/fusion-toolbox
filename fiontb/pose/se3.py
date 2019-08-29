"""Lie algebra operators for SE3 group.
"""
import math

import torch

from fiontb.pose.so3 import hat, vee

# pylint: disable=invalid-name

_I3 = {
    torch.float32: torch.eye(3, dtype=torch.float32),
    torch.float64: torch.eye(3, dtype=torch.float64)
}


class SE3Box(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        dtype = input.dtype

        omega = input[3:]
        omega_skew = hat(omega)
        omega_skew_sq = omega_skew @ omega_skew

        theta = omega.norm()
        theta_sq = theta*theta

        so3 = torch.eye(3, dtype=dtype, device=device)
        so3 += (torch.sin(theta) / theta)*omega_skew
        so3 += ((1.0 - torch.cos(theta)) / theta_sq)*omega_skew_sq

        theta_cu = theta_sq*theta

        V = torch.eye(3)
        V += (1 - torch.cos(theta)) / theta_sq * omega_skew
        V += (theta - torch.sin(theta)) / theta_cu * omega_skew

        se3 = torch.eye(4)
        se3[:3, :3] = so3

        rho = input[:3]
        se3[:3, 3] = V@rho

        return se3

    @staticmethod
    def backward(ctx, grad_output):
        import ipdb; ipdb.set_trace()

        J = torch.tensor()
        pass


def exp(twist):
    omega_vec = twist[3:]
    theta = omega_vec.norm()

    # TODO: handle when theta is zero
    omega = hat(omega_vec)
    omega_sq = omega @ omega

    se3 = torch.zeros(4, 4, dtype=twist.dtype)

    I = _I3[twist.dtype]
    se3[:3, :3] = (I
                   + (math.sin(theta) / theta) * omega
                   + ((1.0 - math.cos(theta)) / (theta*theta)) * omega_sq)

    theta_sq = theta*theta
    theta_cu = theta_sq*theta
    V = (I
         + ((1 - math.cos(theta)) / (theta_sq)) * omega
         + ((theta - math.sin(theta)) / theta_cu) * omega_sq)

    se3[:3, 3] = V @ twist[:3]
    se3[3, 3] = 1.0

    return se3


def log(se3_matrix):
    I = _I3[se3_matrix.dtype]
    R = se3_matrix[:3, :3]

    cos_theta = (torch.trace(R) - 1)*.5
    theta = math.acos(cos_theta)
    R_Rt = R - R.transpose(1, 0)
    log_R = (theta / (2.0 * math.sin(theta))) * R_Rt

    omega_vec = vee(log_R)
    omega = hat(omega_vec)

    log_R_sq = log_R @ log_R
    V_inv = (
        I
        - 0.5*omega
        + (
            (1 - (theta*math.cos(theta*.5) / (2 * math.sin(theta*.5))))
            / theta*theta
        )*log_R_sq)

    t = se3_matrix[:3, 3]
    t = V_inv@t

    return torch.cat((t, omega_vec))


class Tests:
    def torch_func(self):
        box = SE3Box.apply
        ep = torch.tensor([1, 2, 3, 4, 5, 6.0], requires_grad=True)
        T = torch.eye(4)
        K = torch.ones(3, 3)
            
        n = torch.tensor([1, 2, 3.0, 4])
        p = torch.tensor([1, 2.0, 3, 4])
        print(box(ep))

        forward = torch.dot(box(ep) @ p, n)
        forward.backward()


if __name__ == '__main__':
    import fire

    fire.Fire(Tests)
