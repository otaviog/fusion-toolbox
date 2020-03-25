import unittest
from pathlib import Path

import numpy as np

import tenviz.io
import torch

from slamtb.metrics import closest_points
from slamtb.spatial.trigoctree import TrigOctree

_BUNNY_ROOT = (Path(__file__).parent.absolute().parent.parent.parent
               / 'test-data/bunny')


class TestTrigOctree(unittest.TestCase):
    def test_closest_points(self):
        verts, trigs = tenviz.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper_res4.ply')

        np.random.seed(10)
        rand_pts = verts + np.random.rand(*verts.shape)*0.01

        rand_pts[0, :] = rand_pts[197, :]
        octree = TrigOctree(torch.from_numpy(verts).float(),
                            torch.from_numpy(trigs).long(), 16)
        tree_closest, _ = octree.query_closest_points(
            torch.from_numpy(rand_pts).float())

        closest = closest_points(
            rand_pts, verts, trigs)

        torch.testing.assert_allclose(tree_closest, closest)
