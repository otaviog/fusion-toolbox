import unittest

import numpy as np
from fiontb import to_open3d


class TestDatatypes(unittest.TestCase):
    def test_to_open3D(self):
        points = np.random.rand(1000, 3)
        pcl = to_open3d(points)
