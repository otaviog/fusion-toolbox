"""Test metrics
"""

import unittest
from pathlib import Path

import numpy as np

import shapelab
import shapelab.io

import fiontb.metrics

_BUNNY_ROOT = Path(__file__).parent.parent.parent / 'test-data/bunny'


def _view_closest_points(closest, verts, trigs):
    context = shapelab.RenderContext(640, 480)

    mesh = context.add_mesh(verts, trigs)
    mesh.set_polygon_mode(shapelab.PolygonMode.WIREFRAME)

    pcl = context.add_point_cloud(closest)
    pcl.point_size = 2

    view = context.viewer()
    while True:
        key = view.wait_key()
        if key < 0:
            break


class TestMetrics(unittest.TestCase):
    def test_mesh_accuracy(self):
        hq_verts, hq_trigs = shapelab.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper.ply')
        lq2_verts, _ = shapelab.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper_res2.ply')
        lq3_verts, _ = shapelab.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper_res3.ply')
        lq4_verts, _ = shapelab.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper_res4.ply')

        for lq_verts in (lq2_verts, lq3_verts, lq4_verts):
            lq_verts += np.random.rand(*lq_verts.shape)*0.01
            lq_closest = fiontb.metrics.closest_points(
                lq_verts, hq_verts, hq_trigs)
            _view_closest_points(lq_closest, hq_verts, hq_trigs)
            print(fiontb.metrics.mesh_accuracy(lq_verts, lq_closest))
