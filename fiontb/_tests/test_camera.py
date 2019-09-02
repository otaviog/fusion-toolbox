import unittest

import numpy as np
import torch

from fiontb.camera import Homogeneous, KCamera, RTCamera, Project


class TestCamera(unittest.TestCase):
    def test_homogeneous(self):
        matrix = np.random.rand(4, 4)
        points = np.random.rand(100, 3)

        points1 = Homogeneous(matrix) @ points

        points2 = np.insert(points, 3, 1, axis=1)
        points2 = matrix @ points2.reshape(-1, 4, 1)
        points2 = np.delete(points2, 3, 1).squeeze()

        np.testing.assert_almost_equal(points1, points2)

    def test_project(self):
        proj = Project.apply
        for dev in ["cpu:0", "cuda:0"]:
            torch.manual_seed(10)
            input = (torch.rand(3, dtype=torch.double, requires_grad=True),
                     torch.tensor([[45.0, 0, 24],
                                   [0, 45, 24]], dtype=torch.double))

            torch.autograd.gradcheck(proj, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)
