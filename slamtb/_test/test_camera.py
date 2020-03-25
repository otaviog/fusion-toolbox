import unittest
from pathlib import Path

import torch
import fire

from slamtb.camera import (KCamera, RTCamera, Project,
                           RigidTransform)
from slamtb.data.ftb import load_ftb


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


class InteractiveTests:
    def project(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from slamtb.frame import FramePointCloud

        dataset = load_ftb(Path(__file__).parent /
                           "../../test-data/rgbd/sample2")
        frame = dataset[0]
        pcl = FramePointCloud.from_frame(frame).unordered_point_cloud(
            world_space=False, compute_normals=False)

        uvs = Project.apply(pcl.points, frame.info.kcam.matrix)
        
        img = np.zeros((frame.rgb_image.shape[0], frame.rgb_image.shape[1]))
        import ipdb; ipdb.set_trace()

        for uv in uvs:
            
            u = int(round(uv[0].item()))
            v = int(round(uv[1].item()))
            f = frame.rgb_image[v, u, 0]

            img[v, u] = f

        plt.figure()
        plt.imshow(frame.rgb_image[:, :, 0])

        plt.figure()
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    fire.Fire(InteractiveTests())
