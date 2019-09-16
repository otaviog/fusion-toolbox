"""Lie algebra operators for SO3 group.
"""

import torch

from fiontb._cfiontb import (so3t_exp_op_forward as _so3t_exp_op_forward,
                             so3t_exp_op_backward as _so3t_exp_op_backward)

# pylint: disable=invalid-name


class SO3tExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, upsilon_omega):
        y_matrices = _so3t_exp_op_forward(upsilon_omega.view(-1, 6))

        ctx.save_for_backward(upsilon_omega, y_matrices)
        return y_matrices

    @staticmethod
    def backward(ctx, dy_matrices):
        upsilon_omega, y_matrices = ctx.saved_tensors

        dx_upsilon_omega = _so3t_exp_op_backward(
            dy_matrices.view(-1, 3, 4), upsilon_omega.view(-1, 6),
            y_matrices.view(-1, 3, 4))

        return dx_upsilon_omega
