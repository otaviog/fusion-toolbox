
from pathlib import Path
from enum import Enum

import torch
import numpy as np
from PIL import Image

import tenviz
from matplotlib.pyplot import get_cmap

_SHADER_DIR = Path(__file__).parent / 'shaders'


class RenderMode(Enum):
    Color = 0
    Confs = 1
    Times = 2


class SurfelRender(tenviz.DrawProgram):
    def __init__(self, surfel_data):
        self.surfel_data = surfel_data

        num_surfels = surfel_data.max_surfel_count
        self.points = tenviz.buffer_empty(num_surfels, 3, tenviz.BType.Float)
        self.normals = tenviz.buffer_empty(num_surfels, 3, tenviz.BType.Float)
        self.colors = tenviz.buffer_empty(
            num_surfels, 3, tenviz.BType.Uint8, normalize=True)
        self.radii = tenviz.buffer_empty(num_surfels, 1, tenviz.BType.Float)
        self.confs = tenviz.buffer_empty(num_surfels, 1, tenviz.BType.Float)

        super(SurfelRender, self).__init__(tenviz.DrawMode.Points,
                                           vert_shader=_SHADER_DIR / "surfel.vert",
                                           frag_shader=_SHADER_DIR / "surfel.frag",
                                           geo_shader=_SHADER_DIR / "surfel.geo",
                                           ignore_missing=True)

        self['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        self['in_pos'] = self.points
        self['in_color'] = self.colors
        self['in_normal'] = self.normals
        self['in_radius'] = self.radii
        self['in_conf'] = self.confs
        self._max_conf = 0

        surfel_tex = np.array(Image.open(str(Path(__file__).parent
                                             / "assets/surfel.png")))
        surfel_tex = torch.from_numpy(surfel_tex)

        self['BaseTexture'] = tenviz.tex_from_torch(surfel_tex)

        cmap = get_cmap('viridis', 2048)
        cmap_tensor = torch.tensor([cmap(i) for i in range(2048)])[:, :3]
        cmap_tensor = cmap_tensor.view(1, -1, 3)

        self['ColorMap'] = tenviz.tex_from_torch(cmap_tensor,
                                                 target=tenviz.GLTexTarget.k2D)

        self.set_render_mode(RenderMode.Color)

        update_idxs = torch.arange(0, num_surfels, dtype=torch.int64)
        self.update(update_idxs)

    def update(self, update_idxs):
        update_idxs = update_idxs.to(self.surfel_data.device)
        points = self.surfel_data.points[update_idxs]
        self.set_bounds(points)

        self.points[update_idxs] = points
        self.normals[update_idxs] = self.surfel_data.normals[update_idxs]
        self.colors[update_idxs] = self.surfel_data.colors[update_idxs]
        self.radii[update_idxs] = self.surfel_data.radii[update_idxs]
        self.confs[update_idxs] = self.surfel_data.confs[update_idxs]
        self._max_conf = max(
            self.surfel_data.confs[update_idxs].max(), self._max_conf)
        self['MaxConf'] = torch.Tensor([self._max_conf])

        active_idxs = self.surfel_data.get_active_indices()

        self.indices.from_tensor(active_idxs.int())

    def set_render_mode(self, render_mode):
        self['RenderMode'] = torch.tensor([render_mode.value], dtype=torch.int)


def _test():
    import tenviz.io
    from fiontb.fusion.surfel import SurfelData

    geo = tenviz.io.read_3dobject(
        Path(__file__).parent / "../../test-data/bunny/bun_zipper_res4.ply").torch()
    normals = tenviz.geometry.compute_normals(geo.verts, geo.faces)

    device = "cuda:0"

    surfels = SurfelData(geo.verts.size(0), device)

    surfels.points[:] = geo.verts
    surfels.normals[:] = normals

    colors = (torch.abs(torch.rand(geo.verts.size(0), 3))*255).byte()
    surfels.colors[:] = colors

    surfels.radii[:] = torch.rand(
        geo.verts.size(0), dtype=torch.float32).abs()*0.01

    surfels.confs[:] = torch.rand(geo.verts.size(0), dtype=torch.float32).abs()

    visible_indices = torch.arange(
        0, geo.verts.size(0), dtype=torch.int64)

    surfels.mark_active(visible_indices)

    ctx = tenviz.Context(640, 480)
    with ctx.current():
        surfel_render = SurfelRender(surfels)
        # surfel_render.set_render_mode(RenderMode.Confs)
        surfel_render.update(visible_indices)

    viewer = ctx.viewer([surfel_render], tenviz.CameraManipulator.WASD)

    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break


if __name__ == '__main__':
    _test()
