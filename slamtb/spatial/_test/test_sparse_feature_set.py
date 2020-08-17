"""Tests the sparse feature set
"""
import unittest
import io

import torch

from slamtb.spatial.sparse_feature_set import SparseFeatureSet


class TestSparse(unittest.TestCase):
    """Tests the sparse feature set
    """

    def setUp(self):
        """Set up a test point set"""
        yx = torch.tensor([[6, 7],
                           [10, 8],
                           [14, 22],
                           [18, 13],
                           [23, 21]], dtype=torch.int32)
        features = torch.rand(5, 16, dtype=torch.float)
        mask = torch.ones(25, 25, dtype=torch.bool)
        mask[1, 1] = False
        mask[5, 5] = False
        mask[10, 9] = False
        mask[19, 0] = False
        # index = 179, 208, 340, 544 and 560
        self.sparse_set = SparseFeatureSet(yx, features, mask)

    def test_init(self):
        """Verifies the constructors"""
        dict_ver = self.sparse_set.as_dict()
        self.assertEqual([179, 208, 340, 544, 560], list(dict_ver.keys()))

    def test_merge(self):
        """Verify merging two sparse feature set"""
        mask = torch.ones(25, 25, dtype=torch.bool)
        mask[1, 1] = False
        mask[5, 5] = False
        mask[10, 9] = False
        mask[19, 0] = False
        # point_ids = 179, 208, 366, 560, 570
        other_sparse_set = SparseFeatureSet(
            torch.tensor([[6, 7],  # Hit
                          [10, 8],  # Hit
                          [14, 22],  # Hit
                          [19, 14],  # Miss
                          [24, 22]],  # Miss,
                         dtype=torch.int32),
            torch.rand(5, 16, dtype=torch.float),
            mask)

        merge_corresp = torch.tensor(
            [[179, 179],
             [208, 208],
             [340, 366]],
            dtype=torch.int64)
        self.sparse_set.merge(merge_corresp, other_sparse_set)
        dict_ver = self.sparse_set.as_dict()

        self.assertEqual(2.0, dict_ver[179][0])
        self.assertEqual(2.0, dict_ver[208][0])
        self.assertEqual(2.0, dict_ver[340][0])
        self.assertEqual(1.0, dict_ver[544][0])
        self.assertEqual(1.0, dict_ver[560][0])

        self.assertEqual(5, len(dict_ver))

    def test_intern_merge(self):
        """Verify merging intern features"""

        self.sparse_set.merge(
            torch.tensor([[179, 208],
                          [340, 544]], dtype=torch.int64))
        self.assertEqual(3, len(self.sparse_set))
        
        dict_ver = self.sparse_set.as_dict()        
        self.assertTrue(179 in dict_ver)
        self.assertTrue(340 in dict_ver)
        self.assertTrue(560 in dict_ver)
        
            
    def test_add(self):
        """Test adding.
        """
        mask = torch.ones(25, 25, dtype=torch.bool)
        other_sparse_set = SparseFeatureSet(
            torch.tensor([[12, 14],  # 202
                          [8, 2],  # 314
                          [24, 21]],  # 621
                         dtype=torch.int32),
            torch.rand(7, 16, dtype=torch.float), mask)

        dense_ids = torch.arange(25*25, dtype=torch.int64) + 500
        self.sparse_set.add(dense_ids, other_sparse_set)

        dict_ver = self.sparse_set.as_dict()

        self.assertEqual(8, len(dict_ver))
        self.assertTrue(560 in dict_ver)
        self.assertTrue(862 in dict_ver)
        self.assertTrue(1049 in dict_ver)


    def test_serialization(self):
        """Tests the serialization and deserialization for pickling it.
        """
        file = io.BytesIO()
        torch.save(self.sparse_set, file)
        file.seek(0, 0)
        new = torch.load(file).as_dict()
        old = self.sparse_set.as_dict()

        self.assertEqual(old.keys(), new.keys())
if __name__ == '__main__':
    unittest.main()
