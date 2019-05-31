import unittest
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import fiontb.filtering



class TestFiltering(unittest.TestCase):
    def test_blur_depth_image(self):
        depth = cv2.imread(str(Path(__file__).parent / "assets" / "frame_depth.png"),
                           cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(str(Path(__file__).parent / "assets" / "frame_mask.png"),
                          cv2.IMREAD_ANYDEPTH)

        #import ipdb; ipdb.set_trace()

        filter_depth = fiontb.filtering.blur_depth_image(depth, 3, mask)
        plt.figure()        
        plt.imshow(filter_depth)
        plt.figure()
        plt.imshow(depth)
        plt.show()
