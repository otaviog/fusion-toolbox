"""Lie algebra utilites SO3 group.
"""

import torch

from fiontb._cfiontb import (
    ExpRtToMatrixOp as _ExpRtToMatrixOp,
    ExpRtTransformOp as _ExpRtTransformOp,
    QuatRtTransformOp as _QuatRtTransformOp)

# pylint: disable=invalid-name

class ExpRtToMatrix(torch.autograd.Function):
    """Differantiable layer for converting exponential rotation vector and
    a translation vector into a 4x4 matrix.

    The forward and backward passes are implemented in C++.
    """

    @staticmethod
    def forward(ctx, exp_rot_t):
        """Forward pass to compute the matrix [exp(w) t]

        """
        y_matrix = _ExpRtToMatrixOp.forward(exp_rot_t.view(-1, 6).cpu())

        ctx.save_for_backward(exp_rot_t, y_matrix)
        return y_matrix

    @staticmethod
    def backward(ctx, dy_matrix):
        """Computes wrt
        """
        exp_rot_t, y_matrix = ctx.saved_tensors
        #import pdb; pdb.set_trace()

        dx_exp_rot_t = _ExpRtToMatrixOp.backward(
            dy_matrix.view(-1, 3, 4), exp_rot_t.view(-1, 6),
            y_matrix.view(-1, 3, 4))

        return dx_exp_rot_t.to(dy_matrix.device)


def vee(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    vec = vector.view(3)
    output = vec.new().resize_(3, 3).fill_(0)
    output[0, 1] = -vec[2]
    output[1, 0] = vec[2]
    output[0, 2] = vec[1]
    output[2, 0] = -vec[1]
    output[1, 2] = -vec[0]
    output[2, 1] = vec[0]

    return output


class ExpRtTransform(torch.autograd.Function):
    """Differantiable layer that transforms 3D vectors by a exponential
    rotation and a translation vectors.

    The forward and backward passes are implemented in C++.

    """

    @staticmethod
    def forward(ctx, exp_rot_t, points):
        y_points = torch.empty(
            points.size(), dtype=points.dtype, device=points.device)
        _ExpRtTransformOp.forward(exp_rot_t, points, y_points)

        ctx.save_for_backward(exp_rot_t, points)
        return y_points

    @staticmethod
    def backward(ctx, dy_points):
        x_exp_rot_t, x_points = ctx.saved_tensors

        d_exp_rt_loss = torch.empty(
            x_points.size(0), 6, dtype=x_exp_rot_t.dtype, device=x_exp_rot_t.device)
        _ExpRtTransformOp.backward(
            x_exp_rot_t, x_points, dy_points, d_exp_rt_loss)

        return d_exp_rt_loss, None
