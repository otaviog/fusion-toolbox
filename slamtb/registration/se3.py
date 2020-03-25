"""Lie algebra utilites SO3 group.
"""

import torch

from slamtb._cslamtb import (
    ExpRtToMatrixOp as _ExpRtToMatrixOp,
    MatrixToExpRtOp as _MatrixToExpRtOp)

# pylint: disable=invalid-name, arguments-differ


class ExpRtToMatrix(torch.autograd.Function):
    """Differantiable layer for converting translation vector and
    exponential rotation (6-D) into a [3x4] rigid transformation matrix.

    The exponential rotation encodes the angle on its norm.

    The derivation of the formula was done by Gallego, Guillermo, and
    Anthony Yezzi. "A compact formula for the derivative of a 3-D
    rotation in exponential coordinates." Journal of Mathematical
    Imaging and Vision 51, no. 3 (2015): 378-384.

    The forward and backward passes are implemented in C++. Based
    initially on the implementation from Byravan, Arunkumar, Felix
    Leeb, Franziska Meier, and Dieter Fox. "Se3-pose-nets: Structured
    deep dynamics models for visuomotor planning and control." arXiv
    preprint arXiv:1710.00489 (2017).
    """

    @staticmethod
    def forward(ctx, exp_rt):
        """
        Forward pass to compute the matrix
        :math:`\begin{bmatrix}exp(w)|t\\end{bmatrix}`.

        Args:

            exp_rt (obj:`torch.Tensor`): XYZ translation and
             exponential rotation axis. [Bx6] or [6] tensor.

        Returns: (obj:`torch.Tensor`): The rigid transformation
         matrix, [Bx3x4] or [3x4].

        """

        batch = 1 if exp_rt.dim() == 1 else exp_rt.size(0)
        matrix = torch.empty(batch, 3, 4, dtype=exp_rt.dtype,
                             device=exp_rt.device)

        _ExpRtToMatrixOp.forward(exp_rt.view(-1, 6), matrix)

        ctx.save_for_backward(exp_rt, matrix)
        if exp_rt.dim() == 1:
            return matrix.squeeze(0)

        return matrix

    @staticmethod
    def backward(ctx, d_matrix_loss):
        """Computes the gradient of exp_rt w.r.t to the loss.

        Args:

            d_matrix_loss (:obj:`torch.Tensor`): The matrix gradient
             w.r.t. to the loss. Should be a [Bx3x4] matrix.

        Returns: (:obj:`torch.Tensor`):

            The gradient of XYZ translation and exponential map w.r.t
             to the loss.

        """
        exp_rt, y_matrix = ctx.saved_tensors

        batch = 1 if exp_rt.dim() == 1 else exp_rt.size(0)
        d_exp_rt_loss = torch.empty(
            batch, 6, dtype=exp_rt.dtype,
            device=exp_rt.device)

        _ExpRtToMatrixOp.backward(
            d_matrix_loss.view(-1, 3, 4),
            exp_rt.view(-1, 6),
            y_matrix.view(-1, 3, 4),
            d_exp_rt_loss)

        if exp_rt.dim() == 1:
            return d_exp_rt_loss.squeeze(0)
        return d_exp_rt_loss


class MatrixToExpRt(torch.autograd.Function):
    """Converts a rigid transformation matrix into XYZ translation and
    exponential rotation.

    """
    @staticmethod
    def forward(ctx, matrix):
        """Forward pass.

        Args:

            matrix (:obj:`torch.Tensor`): Rigid transformation
             matrix. [Bx3x4] matrix.

        Returns: (:obj:`torch.Tensor`):

            XYZ translation and exponential axis angle rotation
             (rodrigues formula). [Bx6] or [6] vector.

        """

        batch = 1 if matrix.dim() == 2 else matrix.size(0)
        exp_rt = torch.empty(batch, 6, dtype=matrix.dtype,
                             device=matrix.device)
        _MatrixToExpRtOp.forward(matrix.view(-1, 3, 4), exp_rt)

        if matrix.dim() == 2:
            return exp_rt.squeeze(0)

    @staticmethod
    def backward(ctx, d_exp_rt_loss):
        """Not implemented, should compute the gradient of the returned
        matrix w.r.t. to the loss.

        Args:

            d_exp_rt_loss (:obj:`torch.Tensor`): The gradient of
             coordinate w.r.t. the loss function.

        Returns:

        """
        raise NotImplementedError()
