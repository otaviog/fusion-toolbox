import unittest

import numpy as np

from fiontb.camera import Homogeneous, KCamera, RTCamera


class TestCamera(unittest.TestCase):
    def _test_kcamera(self):
        cam = KCamera.create_from_params()

    def _test_rtcamera(self):
        cam = RTcamera.create_from_params()

    def test_homogeneous(self):
        matrix = np.random.rand(4, 4)
        points = np.random.rand(100, 3)

        points1 = Homogeneous(matrix) @ points

        points2 = np.insert(points, 3, 1, axis=1)
        points2 = matrix @ points2.reshape(-1, 4, 1)
        points2 = np.delete(points2, 3, 1).squeeze()

        np.testing.assert_almost_equal(points1, points2)
