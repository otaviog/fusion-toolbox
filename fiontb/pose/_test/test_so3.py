import unittest

import torch

import fiontb.pose.so3 as so3

from ._data import get_rand_se3_mat


class TestSO3(unittest.TestCase):
    def test_vee_hat(self):
        mtx = get_rand_se3_mat()[:3, :3]
        import ipdb; ipdb.set_trace()

        vee = so3.vee(mtx)
        hat = so3.hat(vee)
        torch.testing.assert_allclose(mtx, hat)
