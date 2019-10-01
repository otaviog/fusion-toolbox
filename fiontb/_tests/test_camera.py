import unittest

import torch

from fiontb.camera import (KCamera, RTCamera, Project,
                           RigidTransform)


class TestCamera(unittest.TestCase):
    def test_rigid_transform(self):
        matrix = torch.rand(4, 4)
        points = torch.rand(100, 4)
        points[:, 3] = 1

        ref_result = matrix @ points.view(-1, 4, 1)
        ref_result = ref_result.squeeze()[:, :3]

        result = RigidTransform(matrix) @ points[:, :3]

        torch.testing.assert_allclose(ref_result, result)

        result = points[:, :3].clone()
        result = RigidTransform(matrix).inplace(result)
        torch.testing.assert_allclose(ref_result, result)

    def test_project(self):
        proj = Project.apply
        for dev in ["cpu:0", "cuda:0"]:
            torch.manual_seed(10)
            input = (torch.rand(3, dtype=torch.double, requires_grad=True),
                     torch.tensor([[45.0, 0, 24],
                                   [0, 45, 24]], dtype=torch.double))

            torch.autograd.gradcheck(proj, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)
