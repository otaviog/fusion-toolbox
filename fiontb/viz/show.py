import torch

import tenviz


def show_surfels(surfels_list, width=640, height=480):
    ctx = tenviz.Context(width, height)

    scene = []
    for surfel_cloud in surfels_list:
        surfels = ctx.add_surfel_cloud()

        with ctx.current():
            surfels.points.from_tensor(surfel_cloud.points)
            surfels.update_bounds(surfel_cloud.points)
            surfels.normals.from_tensor(surfel_cloud.normals)

            surfels.colors.from_tensor(surfel_cloud.colors)
            radii = torch.full((surfel_cloud.size,), 0.01, dtype=torch.float)
            surfels.radii.from_tensor(radii)
            surfels.mark_visible(torch.arange(0, surfel_cloud.size))

        scene.append(surfels)

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
