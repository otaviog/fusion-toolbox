import unittest

from scipy.spatial import cKDTree

import torch
from fiontb.sparse.octtree import OctTree

class TestOctTree(unittest.TestCase):
    def test_query(self):
        points = torch.rand(100, 3)
        
        octtree = OctTree(points)
        
        dists, idxs = octtree.query(points[0:5], 0.1, 5)
        pass
    
