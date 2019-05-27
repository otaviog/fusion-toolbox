import unittest

import torch
import torch.testing as torcht

from fiontb.sparse.aabb import AABB, subdivide_octo


class TestAABB(unittest.TestCase):
    def test_subdivide_octo(self):
        box = AABB((-10, -10, -10), (10, 10, 10))

        boxes = subdivide_octo(box)

        self.assertEqual(8, len(boxes))
        # Left-Bottom-Near
        torcht.assert_allclose(boxes[0].min, torch.Tensor([-10., -10., -10.]))
        torcht.assert_allclose(boxes[0].max, torch.Tensor([0., 0., 0.]))

        # Left-Bottom-Far
        torcht.assert_allclose(boxes[1].min, torch.Tensor([-10., -10., 0.]))
        torcht.assert_allclose(boxes[1].max, torch.Tensor([0., 0., 10.]))

        # Left-Top-Far
        torcht.assert_allclose(boxes[2].min, torch.Tensor([-10., 0., 0.]))
        torcht.assert_allclose(boxes[2].max, torch.Tensor([0., 10., 10.]))

        # Left-Top-Near
        torcht.assert_allclose(boxes[3].min, torch.Tensor([-10., 0., -10.]))
        torcht.assert_allclose(boxes[3].max, torch.Tensor([0., 10., 0.]))

        # Right-Bottom-Near
        torcht.assert_allclose(boxes[4].min, torch.Tensor([0., -10., -10.]))
        torcht.assert_allclose(boxes[4].max, torch.Tensor([10., 0., 0.]))

        # Right-Bottom-Far
        torcht.assert_allclose(boxes[5].min, torch.Tensor([0., -10., 0.]))
        torcht.assert_allclose(boxes[5].max, torch.Tensor([10., 0., 10.]))

        # Right-Top-Far
        torcht.assert_allclose(boxes[6].min, torch.Tensor([0., 0., 0.]))
        torcht.assert_allclose(boxes[6].max, torch.Tensor([10., 10., 10.]))

        # Right-Top-Near
        torcht.assert_allclose(boxes[7].min, torch.Tensor([0., 0., -10.]))
        torcht.assert_allclose(boxes[7].max, torch.Tensor([10., 10., 0.]))

    def test_isinside(self):
        box = AABB((-10, -10, -10), (10, 10, 10))
        inside_mask = box.is_inside(
            torch.Tensor([[0, 0, 0],
                          [6, 7, 2],
                          [-10, -10, -10],
                          [-10, 0, 5],
                          [15, 14, 13],
                          [15, 8, 7],
                          [15, 8, 13],
                          [5, 12, 3],
                          [15, 10, 10]]))

        torcht.assert_allclose(torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.uint8),
                               inside_mask)
        print(inside_mask)
