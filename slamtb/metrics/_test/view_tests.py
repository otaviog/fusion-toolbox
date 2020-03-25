from pathlib import Path

import torch
import fire

import tenviz
import tenviz.io

_BUNNY_ROOT = Path(__file__).parent.absolute(
).parent.parent / 'test-data/bunny'


def _view_pcl_mesh(rand_pts, points, verts, trigs):
    context = tenviz.Context(640, 480)

    with context.current():
        mesh = tenviz.nodes.create_mesh(torch.from_numpy(
            verts).float(), torch.from_numpy(trigs))
        mesh.style.polygon_mode = tenviz.PolygonMode.Wireframe

        colors = (torch.rand(rand_pts.shape).abs()*255).byte()

        rands_pcl = tenviz.create_point_cloud(
            torch.from_numpy(rand_pts).float(), colors)
        rands_pcl.style.point_size = 4

        close_pcl = tenviz.create_point_cloud(points, colors)
        close_pcl.style.point_size = 4

    view = context.viewer([mesh, rands_pcl, close_pcl])
    while True:
        key = view.wait_key(1)
        if key < 0:
            break


class _Testing:

    @staticmethod
    def closest_points():
        verts, trigs = tenviz.io.read_3dobject(
            _BUNNY_ROOT / 'bun_zipper_res4.ply')

        rand_pts = verts + np.random.rand(*verts.shape)*0.01
        closest = slamtb.metrics.closest_points(
            rand_pts, verts, trigs)
        _view_pcl_mesh(rand_pts, closest, verts, trigs)

    @staticmethod
    def sample_points():
        sample_points = slamtb.metrics.sample_points(verts, trigs, 50)
        _view_pcl_mesh(sample_points, verts, trigs)


if __name__ == '__main__':
    Fire(_Testing)
