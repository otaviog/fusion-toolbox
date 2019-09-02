"""Lie algebra operators for SE3 group.
"""
import math

import torch

from fiontb.pose.so3 import hat, vee
from fiontb._cfiontb import (se3_exp_op_forward as _se3_exp_op_forward,
                             se3_exp_op_backward as _se3_exp_op_backward)

# pylint: disable=invalid-name

_I3 = {
    torch.float32: torch.eye(3, dtype=torch.float32),
    torch.float64: torch.eye(3, dtype=torch.float64)
}


class SE3Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, upsilon_omega):
        y_matrices = _se3_exp_op_forward(upsilon_omega.view(-1, 1))

        ctx.save_for_backward(upsilon_omega, y_matrices)

        return y_matrices

    @staticmethod
    def backward(ctx, dy_matrices):
        upsilon_omega, y_matrices = ctx.saved_tensors
        dx_upsilon_omega = _se3_exp_op_backward(
            dy_matrices.view(-1, 3, 3), upsilon_omega, y_matrices)
        return dx_upsilon_omega


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
