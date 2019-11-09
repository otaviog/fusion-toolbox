import torch

import tenviz

from fiontb.frame import FramePointCloud
from fiontb.pointcloud import PointCloud
from fiontb.surfel import SurfelCloud, SurfelModel


def show_pcls(pcl_list, width=640, height=480, overlay_mesh=None, point_size=1):
    ctx = tenviz.Context(width, height)

    with ctx.current():
        scene = []
        for pcl in pcl_list:
            tv_pcl = tenviz.create_point_cloud(pcl.points.view(-1, 3),
                                               pcl.colors.view(-1, 3))
            tv_pcl.style.point_size = int(point_size)
            scene.append(tv_pcl)

        if overlay_mesh is not None:
            mesh = tenviz.create_mesh(overlay_mesh.verts, overlay_mesh.faces,
                                      overlay_mesh.normals)
            mesh.style.polygon_mode = tenviz.PolygonMode.Wireframe
            scene.append(mesh)

    viewer = ctx.viewer(scene, tenviz.CameraManipulator.WASD)
    viewer.reset_view()
    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break

        key = chr(key & 0xff)
        if '1' <= key <= '9':
            toggle_idx = int(key) - 1
            if toggle_idx < len(scene):
                scene[toggle_idx].visible = not scene[toggle_idx].visible


def geoshow(geometries, width=640, height=480, point_size=3):

    if isinstance(geometries, list):
        if not geometries:
            return

    if not isinstance(geometries, list):
        geometries = [geometries]

    ctx = tenviz.Context(width, height)

    scene = []
    with ctx.current():
        for geom in geometries:
            if isinstance(geom, (FramePointCloud, PointCloud)):
                if isinstance(geom, FramePointCloud):
                    geom = geom.unordered_point_cloud(
                        world_space=False, compute_normals=False)

                pcl = tenviz.create_point_cloud(geom.points.view(-1, 3),
                                                geom.colors.view(-1, 3))
                pcl.style.point_size = int(point_size)
                scene.append(pcl)

    viewer = ctx.viewer(scene, tenviz.CameraManipulator.WASD)
    viewer.reset_view()
    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break

        key = chr(key & 0xff)
        if '1' <= key <= '9':
            toggle_idx = int(key) - 1
            if toggle_idx < len(scene):
                scene[toggle_idx].visible = not scene[toggle_idx].visible
