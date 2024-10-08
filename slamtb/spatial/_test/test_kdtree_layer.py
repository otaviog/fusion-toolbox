"""Test the diffentiable KDTree layer.
"""
import unittest

import torch

from slamtb.spatial.kdtree_layer import KDTreeLayer


class TestKDTreeLayer(unittest.TestCase):
    """Test the diffentiable KDTree layer.
    """

    @staticmethod
    def test_gradcheck():
        """Gradcheck
        """
        torch.manual_seed(10)
        source_xyz = torch.rand(20, 3, dtype=torch.double, requires_grad=True)
        target_xyz = torch.rand(100, 3, dtype=torch.double)
        features = torch.rand(5, 100, dtype=torch.double)

        layer = KDTreeLayer.setup(target_xyz)

        # pylint: disable=protected-access
        KDTreeLayer._gradcheck = True

        source_xyz = torch.rand(20, 3, dtype=torch.double, requires_grad=True)

        inputs = (features, source_xyz)
        torch.autograd.gradcheck(layer, inputs, eps=1e-6, atol=1e-4,
                                 raise_exception=True)
