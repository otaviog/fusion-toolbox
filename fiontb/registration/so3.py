"""Lie algebra utilites SO3 group.
"""

import torch

from fiontb._cfiontb import (SO3tExpOp as _SO3tExpOp)

# pylint: disable=invalid-name


class SO3tExp(torch.autograd.Function):
    """Hack operator for differentiable SO3 rotation and a
    translation. Obsivly this should be a SE3 operator, but we're
    unable to correct implement the SE3 grad.

    """

    @staticmethod
    def forward(ctx, upsilon_omega):
        upsilon_omega = upsilon_omega.cpu()
        y_matrices = _SO3tExpOp.forward(upsilon_omega.view(-1, 6).cpu())

        ctx.save_for_backward(upsilon_omega, y_matrices)
        return y_matrices

    @staticmethod
    def backward(ctx, dy_matrices):
        upsilon_omega, y_matrices = ctx.saved_tensors
        dx_upsilon_omega = _SO3tExpOp.backward(
            dy_matrices.view(-1, 3, 4), upsilon_omega.view(-1, 6),
            y_matrices.view(-1, 3, 4))

        return dx_upsilon_omega.to(dy_matrices.device)
