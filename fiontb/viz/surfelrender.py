from pathlib import Path
from enum import Enum

import torch
import numpy as np
from matplotlib.pyplot import get_cmap
import tenviz

from fiontb.surfel import SurfelModel, SurfelCloud

_SHADER_DIR = Path(__file__).parent / 'shaders'


class RenderMode(Enum):
    Color = 0
    Confs = 1
    Normal = 2
    Gray = 3
    Times = 4
    Ids = 5


class SurfelRender(tenviz.DrawProgram):
    def __init__(self, surfel_model):
        self.surfel_model = surfel_model

        with surfel_model.gl_context.current():
            super(SurfelRender, self).__init__(tenviz.DrawMode.Points,
                                               vert_shader_file=_SHADER_DIR / "surfel.vert",
                                               frag_shader_file=_SHADER_DIR / "surfel.frag",
                                               geo_shader_file=_SHADER_DIR / "surfel.geom",
                                               ignore_missing=True)

            self['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            self['in_pos'] = surfel_model.positions
            self['in_normal'] = surfel_model.normals
            self['in_color'] = surfel_model.colors
            self['in_radius'] = surfel_model.radii
            self['in_conf'] = surfel_model.confidences
            self['in_time'] = surfel_model.times
            self['in_mask'] = surfel_model.free_mask_gl
            self._max_conf = 0

            cmap = get_cmap('inferno', 2048)
            cmap_tensor = torch.tensor([cmap(i) for i in range(2048)])[:, :3]
            cmap_tensor = cmap_tensor.view(1, -1, 3)

            self['ColorMap'] = tenviz.tex_from_tensor(
                cmap_tensor, target=tenviz.TexTarget.k2D)

        self.set_stable_threshold(-1.0)
        self.set_render_mode(RenderMode.Color)
        # self.set_bounds(torch.tensor([[0, 0, 0],
        #                              [100, 100, 100]], dtype=torch.float32))

    def set_render_mode(self, render_mode):
        with self.surfel_model.gl_context.current():
            self['RenderMode'] = int(render_mode.value)

        self.render_mode = render_mode

    def set_max_confidence(self, max_conf):
        with self.surfel_model.gl_context.current():
            self['MaxConf'] = float(max_conf)

    def set_max_time(self, max_time):
        with self.surfel_model.gl_context.current():
            self['MaxTime'] = float(max_time)

    def set_stable_threshold(self, stable_conf_thresh):
        with self.surfel_model.gl_context.current():
            self['StableThresh'] = float(stable_conf_thresh)


def show_surfels(gl_context, surfels_list, title="Surfels",
                 max_time=10, max_conf=10, view_matrix=None, invert_y=True):
    from fiontb.fusion.surfel.indexmap import SurfelIndexMapRaster

    scene = []

    transform = np.eye(4)
    if invert_y:
        transform[1, 1] = -1

    for surfels in surfels_list:
        if isinstance(surfels, SurfelCloud):
            node = SurfelRender(
                SurfelModel.from_surfel_cloud(gl_context, surfels))
        elif isinstance(surfels, SurfelModel):
            node = SurfelRender(surfels)
        else:
            print("Invalid instance")
            continue

        node.transform = transform
        scene.append(node)

    viewer = gl_context.viewer(scene, tenviz.CameraManipulator.WASD)

    raster = SurfelIndexMapRaster(scene[0].surfel_model)

    if view_matrix is not None:
        viewer.view_matrix = view_matrix.clone().numpy()
    else:
        viewer.reset_view()
    viewer.title = title

    print("""Keys:
    r: Show colors
    t: Show normals
    y: Show confidences
    u: Show times
    o: Show gray
    i: Show Ids
    """)
    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break

        key = chr(key & 0xff).lower()
        if '1' <= key <= '9':
            toggle_idx = int(key) - 1
            if toggle_idx < len(scene):
                scene[toggle_idx].visible = not scene[toggle_idx].visible

        mode_dict = {'y': RenderMode.Confs,
                     'r': RenderMode.Color,
                     't': RenderMode.Normal,
                     'o': RenderMode.Gray,
                     'u': RenderMode.Times,
                     'i': RenderMode.Ids}

        if key in mode_dict:
            for snode in scene:
                snode.set_render_mode(mode_dict[key])
                snode.set_max_time(max_time)
                snode.set_max_confidence(max_conf)

        if key == 'l':
            import matplotlib.pyplot as plt

            raster.raster(torch.from_numpy(viewer.projection_matrix),
                          torch.from_numpy(viewer.view_matrix),
                          viewer.width, viewer.height)
            indexmap = raster.to_indexmap()
            _, ax = plt.subplots()
            indices = indexmap.indexmap.cpu().flip([0]).numpy()[:, :, 0]
            ax.imshow(indices)

            def _format_coord(x, y):
                return "{} {} {}".format(x, y, indices[int(y), int(x)])

            ax.format_coord = _format_coord

            plt.figure()
            plt.imshow(indexmap.color.cpu().flip([0]).numpy())
            plt.show()


def _test():
    import tenviz.io

    geo = tenviz.io.read_3dobject(
        Path(__file__).parent / "../../test-data/bunny/bun_zipper_res4.ply").torch()
    normals = tenviz.geometry.compute_normals(geo.verts, geo.faces)

    nverts = geo.verts.size(0)
    cloud = SurfelCloud(
        geo.verts,
        torch.rand(nverts, dtype=torch.float32).abs(),
        normals,
        torch.rand(nverts, dtype=torch.float32).abs()*0.01,
        (torch.abs(torch.rand(nverts, 3))*255).byte()).to("cuda:0")
    ctx = tenviz.Context(640, 480)
    model = SurfelModel(ctx, geo.verts.size(0))
    model.add_surfels(cloud)
    model.update_gl()

    surfel_render = SurfelRender(model)
    surfel_render.set_bounds(geo.verts)
    # surfel_render.set_render_mode(RenderMode.Confs)

    viewer = ctx.viewer([surfel_render], tenviz.CameraManipulator.WASD)

    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break


if __name__ == '__main__':
    _test()
