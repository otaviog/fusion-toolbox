import unittest

import torch
import numpy as np

from slamtb.registration.correspondence_map import CorrespondenceMap
from slamtb.frame import FramePointCloud
from slamtb.testing import load_sample2_dataset


class TestCorrespondenceMap(unittest.TestCase):
    def test_correspondence_map(self):

        corresp_map = CorrespondenceMap(10, 10)

        dataset = load_sample2_dataset()

        source = FramePointCloud.from_frame(dataset[0])
        target = FramePointCloud.from_frame(dataset[13])

        comaps = []

        for device in ["cpu", "cuda:0"]:
            source = source.to(device)
            target = target.to(device)

            comap = corresp_map(source.points, source.normals, source.mask,
                                torch.eye(4).to(device),
                                target.points, target.normals, target.mask,
                                source.kcam.matrix.to(device))

            comaps.append(comap.cpu().numpy())

        #torch.testing.assert_allclose(np.right_shift(comaps[0], 32),
        #                              np.right_shift(comaps[1], 32))
        __import__("ipdb").set_trace()
        torch.testing.assert_allclose(np.bitwise_and(comaps[0], 0xffffffff),
                                      np.bitwise_and(comaps[1], 0xffffffff))

        
if __name__ == '__main__':
    unittest.main()
