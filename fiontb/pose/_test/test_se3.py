import unittest

import torch

import fiontb.pose.se3 as se3


class TestSE3(unittest.TestCase):
    def test_exp(self):
        # trf = se3.exp(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # torch.testing.assert_allclose(torch.eye(4), trf)

        twist = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.1, 0.2], dtype=torch.float64)
        mtx = se3.exp(twist)
        twist0 = se3.log(mtx)
        import ipdb; ipdb.set_trace()

        torch.testing.assert_allclose(torch.eye(4), trf)
