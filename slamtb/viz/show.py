"""Quick showing
"""
import torch

import tenviz

from slamtb.frame import FramePointCloud
from slamtb.pointcloud import PointCloud
from slamtb.surfel import SurfelCloud, SurfelModel
from .surfelrender import SurfelRender


def create_tenviz_node_from_geometry(geometry, context=None,
                                     point_size=3):
    """Create a TensorViz node from a geometry.

    Geometries are kept in their local coordinates.

    Args:

        geometry (Object): (Almost) any slamtb geometry type.

        context (:obj:`tenviz.Context`): TensorViz context, required
         for the :class:`SurfelCloud` type.

        point_size (int): Point size for point-cloud types.

    Returns: (:obj:`tenviz.Node`):

        Tenviz node object.

    Raises:

        RuntimeError: If it cannot determine the geometry type.

    """
    if isinstance(geometry, (FramePointCloud, PointCloud)):
        pcl = geometry
        if isinstance(geometry, FramePointCloud):
            pcl = geometry.unordered_point_cloud(
                world_space=False, compute_normals=False)

        node = tenviz.nodes.PointCloud(pcl.points.view(-1, 3),
                                       pcl.colors.view(-1, 3))
        node.style.point_size = int(point_size)
    elif isinstance(geometry, (SurfelCloud, SurfelModel)):
        surfels = geometry
        if isinstance(geometry, SurfelCloud):
            if context is None:
                raise RuntimeError(
                    "`context` is required for SurfelCloud type")
            surfels = SurfelModel.from_surfel_cloud(context, geometry)
        node = SurfelRender(surfels)
    elif isinstance(geometry, tenviz.geometry.Geometry):
        node = tenviz.nodes.create_mesh_from_geo(geometry)
    else:
        raise RuntimeError("Unknown geometry type:", type(geometry))

    return node


def geoshow(geometries, width=640, height=480, point_size=3,
            title="slamtb.viz.geoshow", invert_y=False,
            view_matrix=None):
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
            node = create_tenviz_node_from_geometry(
                geom, context=ctx, point_size=point_size)
            node.transform = transform
            scene.append(node)

    viewer = ctx.viewer(scene, tenviz.CameraManipulator.WASD)
    viewer.title = title
    viewer.reset_view()

    if view_matrix is not None:
        viewer.view_matrix = view_matrix

    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break

        key = chr(key & 0xff)
        if '1' <= key <= '9':
            toggle_idx = int(key) - 1
            if toggle_idx < len(scene):
                scene[toggle_idx].visible = not scene[toggle_idx].visible
