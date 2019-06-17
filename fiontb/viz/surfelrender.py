
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
            self._max_conf = 0

            cmap = get_cmap('plasma', 2048)
            cmap_tensor = torch.tensor([cmap(i) for i in range(2048)])[:, :3]
            cmap_tensor = cmap_tensor.view(1, -1, 3)

            self['ColorMap'] = tenviz.tex_from_torch(cmap_tensor,
                                                     target=tenviz.GLTexTarget.k2D)

        self.update()
        self.set_stable_threshold(-1.0)
        self.set_render_mode(RenderMode.Color)

    def update(self):
        active_idxs = self.surfel_model.get_active_indices()
        with self.surfel_model.context.current():
            self.indices.from_tensor(active_idxs.int())

    def set_render_mode(self, render_mode):
        with self.surfel_model.context.current():
            self['RenderMode'] = torch.tensor(
                [render_mode.value], dtype=torch.int)

    def set_max_confidence(self, max_conf):
        with self.surfel_model.context.current():
            self['MaxConf'] = torch.tensor(
                [max_conf], dtype=torch.float)

    def set_max_time(self, max_time):
        with self.surfel_model.context.current():
            self['MaxTime'] = torch.tensor(
                [max_time], dtype=torch.float)

    def set_stable_threshold(self, stable_conf_thresh):
        with self.surfel_model.context.current():
            self['StableThresh'] = torch.tensor(
                [stable_conf_thresh], dtype=torch.float32)


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
                        torch.rand(nverts(0), dtype=torch.float32).abs(),
                        torch.full(nverts, dtype=torch.int32))
    ctx = tenviz.Context(640, 480)
    model = SurfelModel(ctx, geo.verts.size(0))
    model.add_surfels(cloud)
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
