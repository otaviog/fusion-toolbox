import unittest

import torch

import fiontb.pose.se3 as se3


class TestSE3(unittest.TestCase):

    def test_exp_op(self):
        exp = se3.SE3Exp.apply
        torch.manual_seed(10)
        for dev in ["cpu:0", "cuda:0"]:
            input = (torch.rand(1, 6, dtype=torch.double, requires_grad=True, device=dev),)
            torch.autograd.gradcheck(exp, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)
