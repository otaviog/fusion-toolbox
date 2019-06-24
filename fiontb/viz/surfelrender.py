
from pathlib import Path
from enum import Enum

import torch
import numpy as np
from PIL import Image
from matplotlib.pyplot import get_cmap

import tenviz

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

        with surfel_model.context.current():
            super(SurfelRender, self).__init__(tenviz.DrawMode.Points,
                                               vert_shader=_SHADER_DIR / "surfel.vert",
                                               frag_shader=_SHADER_DIR / "surfel.frag",
                                               geo_shader=_SHADER_DIR / "surfel.geom",
                                               ignore_missing=True)

            self['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            self['in_pos'] = surfel_model.points
            self['in_normal'] = surfel_model.normals
            self['in_color'] = surfel_model.colors
            self['in_radius'] = surfel_model.radii
            self['in_conf'] = surfel_model.confs
            self['in_time'] = surfel_model.times
            self['in_mask'] = surfel_model.active_mask_gl
            self._max_conf = 0

            cmap = get_cmap('plasma', 2048)
            cmap_tensor = torch.tensor([cmap(i) for i in range(2048)])[:, :3]
            cmap_tensor = cmap_tensor.view(1, -1, 3)

            self['ColorMap'] = tenviz.tex_from_tensor(
                cmap_tensor, target=tenviz.GLTexTarget.k2D)

        self.set_stable_threshold(-1.0)
        self.set_render_mode(RenderMode.Color)

    def set_render_mode(self, render_mode):
        with self.surfel_model.context.current():
            self['RenderMode'] = int(render_mode.value)

    def set_max_confidence(self, max_conf):
        with self.surfel_model.context.current():
            self['MaxConf'] = float(max_conf)

    def set_max_time(self, max_time):
        with self.surfel_model.context.current():
            self['MaxTime'] = float(max_time)

    def set_stable_threshold(self, stable_conf_thresh):
        with self.surfel_model.context.current():
            self['StableThresh'] = float(stable_conf_thresh)


def _test():
    import tenviz.io
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud

    geo = tenviz.io.read_3dobject(
        Path(__file__).parent / "../../test-data/bunny/bun_zipper_res4.ply").torch()
    normals = tenviz.geometry.compute_normals(geo.verts, geo.faces)

    nverts = geo.verts.size(0)
    cloud = SurfelCloud(geo.verts,
                        (torch.abs(torch.rand(nverts, 3))*255).byte(),
                        normals, torch.rand(
                            nverts, dtype=torch.float32).abs()*0.01,
                        torch.rand(nverts, dtype=torch.float32).abs(),
                        torch.full((nverts, ), 1, dtype=torch.int32), "cpu")
    cloud.to("cuda:0")
    ctx = tenviz.Context(640, 480)
    model = SurfelModel(ctx, geo.verts.size(0))
    model.add_surfels(cloud)
    model.update_active_mask_gl()
    surfel_render = SurfelRender(model)
    surfel_render.set_bounds(geo.verts)
    # surfel_render.set_render_mode(RenderMode.Confs)

    viewer = ctx.viewer([surfel_render], tenviz.CameraManipulator.WASD)

    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break


def show_surfels(context, surfels_list, overlay_mesh=None,
                 title="Surfels", max_time=10, max_conf=10):
    scene = [SurfelRender(surfels) for surfels in surfels_list]
    viewer = context.viewer(scene, tenviz.CameraManipulator.WASD)
    viewer.set_title(title)
    viewer.reset_view()
    print("""Keys:
    I: Show confidences
    U: Show colors
    O: Show normals
    Y: Show times
    P: Show gray
    """)
    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break

        key = chr(key & 0xff)
        if '1' <= key <= '9':
            toggle_idx = int(key) - 1
            if toggle_idx < len(scene):
                scene[toggle_idx].visible = not scene[toggle_idx].visible

        mode_dict = {'I': RenderMode.Confs,
                     'U': RenderMode.Color,
                     'O': RenderMode.Normal,
                     'P': RenderMode.Gray,
                     'Y': RenderMode.Times,
                     'T': RenderMode.Ids}
        if key in mode_dict:
            for snode in scene:
                snode.set_render_mode(mode_dict[key])
                snode.set_max_time(max_time)
                snode.set_max_confidence(max_conf)


if __name__ == '__main__':
    _test()
