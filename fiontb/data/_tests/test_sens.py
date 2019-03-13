#!/usr/bin/env python
"""Tests sens loading and writing
"""

# pylint: disable=missing-docstring, no-self-use

import unittest
from pathlib import Path

import numpy.testing as npt

import fiontb.data.sens as sens

from ._utils import load_scene2

_OUTPUT_FILE = str(Path(__file__).parent / "out-sens.sens")


class TestSens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene2_ds = load_scene2()

    def test_1_write(self):
        sens.write_sens(_OUTPUT_FILE, self.scene2_ds, self.scene2_ds[0])

    def test_2_read(self):
        sens_ds = sens.load_sens(_OUTPUT_FILE)
        self.assertEqual(len(self.scene2_ds), len(sens_ds))

        src_snap = self.scene2_ds[58]
        dst_snap = sens_ds[58]

        npt.assert_almost_equal(src_snap.kcam.matrix, dst_snap.kcam.matrix)
        npt.assert_almost_equal(src_snap.rt_cam.matrix, dst_snap.rt_cam.matrix)

        npt.assert_almost_equal(src_snap.rgb_image, dst_snap.rgb_image)
        npt.assert_almost_equal(src_snap.depth_image, dst_snap.depth_image)


def _main():
    from fiontb.viz.datasetviewer import DatasetViewer

    ds_viewer = DatasetViewer(sens.load_sens(_OUTPUT_FILE), "Test sens")
    ds_viewer.run()


if __name__ == '__main__':
    _main()
