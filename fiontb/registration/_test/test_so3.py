"""Unit testing of SO3 operations
"""
import unittest

import torch

import fiontb.registration.so3 as so3


class TestSO3(unittest.TestCase):
    """Test SO3 class.
    """

    # pylint: disable=no-self-use, redefined-builtin
    def test_exp_op(self):
        """Gradcheck the SO3t EXP operator.

        """
        exp = so3.SO3tExp.apply
        torch.manual_seed(10)
        for dev in ["cpu:0", "cuda:0"]:
            input = (torch.rand(1, 6, dtype=torch.double,
                                requires_grad=True, device=dev),)
            torch.autograd.gradcheck(exp, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)
