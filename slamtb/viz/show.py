"""Quick showing
"""
import torch

import tenviz

from slamtb.frame import FramePointCloud
from slamtb.pointcloud import PointCloud
from slamtb.surfel import SurfelCloud, SurfelModel
from .surfelrender import SurfelRender


def geoshow(geometries, width=640, height=480, point_size=3,
            title="slamtb.viz.geoshow", invert_y=False):
    """Show Fusion Toolbox geometry data types.

    The geometries display can be toggle using number keys starting at '1'.

    Args:

        geometries (List[Any]): Fusion Toolbox geometry list.

        width (int): Window width.

        height (int): Window height.

        point_size (int): Size of points.

        title (str): Window title.

        invert_y (bool): Whatever to flip the scene.

    """
    # pylint: disable=too-many-branches

    if isinstance(geometries, list):
        if not geometries:
            return

    if not isinstance(geometries, list):
        geometries = [geometries]

    ctx = tenviz.Context(width, height)

    transform = torch.eye(4, dtype=torch.float)
    if invert_y:
        transform[1, 1] = -1

    scene = []
    with ctx.current():
        for geom in geometries:
            if isinstance(geom, (FramePointCloud, PointCloud)):
                if isinstance(geom, FramePointCloud):
                    geom = geom.unordered_point_cloud(
                        world_space=False, compute_normals=False)

                node = tenviz.nodes.PointCloud(geom.points.view(-1, 3),
                                               geom.colors.view(-1, 3))
                node.style.point_size = int(point_size)
            elif isinstance(geom, (SurfelCloud, SurfelModel)):
                if isinstance(geom, SurfelCloud):
                    geom = SurfelModel.from_surfel_cloud(ctx, geom)
                node = SurfelRender(geom)

            elif isinstance(geom, tenviz.geometry.Geometry):
                node = tenviz.nodes.create_mesh_from_geo(geom)
            else:
                print("Unknown geometry type:", type(geom))
                continue

            node.transform = transform
            scene.append(node)

    viewer = ctx.viewer(scene, tenviz.CameraManipulator.WASD)
    viewer.title = title
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
