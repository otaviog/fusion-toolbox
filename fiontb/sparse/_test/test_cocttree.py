import unittest

from scipy.spatial import cKDTree

import torch
from fiontb.fiontblib import OctTree


class TestOctTree(unittest.TestCase):
    def test_query(self):
        torch.manual_seed(10)
        points = torch.rand(10000, 3).to("cuda:0")
        octtree = OctTree(points, 5)

        query_points = torch.rand(500, 3).to("cuda:0")
        
        dists, idxs = octtree.query(query_points, 5, 0.1)

        tree = cKDTree(points.cpu().numpy())
        ref_dists, ref_idxs = tree.query(query_points.cpu().numpy(),
                                         5, distance_upper_bound=0.1)
        ref_idxs[ref_idxs == points.size(0)] = -1

        ref_dists = torch.from_numpy(ref_dists).float()
        ref_idxs = torch.from_numpy(ref_idxs).long()

        torch.testing.assert_allclose(ref_idxs, idxs.cpu())
        torch.testing.assert_allclose(ref_dists, dists.cpu())
