"""Unit testing of SO3 operations
"""
import unittest

import torch

import fiontb.registration.se3 as se3


class TestSE3(unittest.TestCase):
    """Test SE3 module.
    """
    # pylint: disable=no-self-use, redefined-builtin

    def test_exp_rt_to_matrix(self):
        """Gradcheck the ExpRtToMatrix operator.
        """
        torch.manual_seed(10)
        for dev in ["cpu:0", "cuda:0"]:
            input = (torch.rand(1, 6, dtype=torch.double,
                                requires_grad=True, device=dev),)
            torch.autograd.gradcheck(
                se3.ExpRtToMatrix.apply,
                input, eps=1e-6, atol=1e-4,
                raise_exception=True)

    def test_matrix_to_exp_rt(self):
        """Test the MatrixToExpRt operator.
        """
        matrix = torch.Tensor([[0.7350, 0.3997, -0.5477, -0.2654],
                               [-0.4502, 0.8917, 0.0464, 0.3553],
                               [0.5069, 0.2125, 0.8354, -0.3161]])

        exp_rt = se3.MatrixToExpRt.apply(matrix)
        matrix2 = se3.ExpRtToMatrix.apply(exp_rt)
        torch.testing.assert_allclose(matrix, matrix2)
