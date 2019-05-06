
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import tenviz


_SHADER_DIR = Path(__file__).parent / 'shaders'


class SurfelCloud(tenviz.DrawProgram):
    def __init__(self, num_surfels):

        self.points = tenviz.buffer_empty(num_surfels, 3, torch.float32)
        self.normals = tenviz.buffer_empty(num_surfels, 3, torch.float32)
        self.colors = tenviz.buffer_empty(num_surfels, 3, torch.uint8)
        self.radii = tenviz.buffer_empty(num_surfels, 1, torch.float)

        super(SurfelCloud, self).__init__(tenviz.DrawMode.Points,
                                          vert_shader=_SHADER_DIR / "surfel.vert",
                                          frag_shader=_SHADER_DIR / "surfel.frag",
                                          geo_shader=_SHADER_DIR / "surfel.geo")

        self['Modelview'] = tenviz.MatPlaceholder.Modelview
        self['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        self['vertex_pos'] = self.points
        self['vertex_color'] = self.colors
        self['vertex_normal'] = self.normals
        self['radii'] = self.radii

        surfel_tex = np.array(Image.open(str(Path(__file__).parent
                                             / "assets/surfel.png")))
        surfel_tex = torch.from_numpy(surfel_tex)

        self['BaseTexture'] = tenviz.tex_from_torch(surfel_tex)
        self.draw_idx_set = set()

    def update(self, surfels, update_idxs, removal_idxs):
        update_idxs_cpu = update_idxs
        update_idxs = update_idxs.to(surfels.device)

        points = surfels.points[update_idxs]
        self.update_bounds(points)

        self.points[update_idxs] = points
        self.normals[update_idxs] = surfels.normals[update_idxs]
        colors = surfels.colors[update_idxs].byte()
        self.colors[update_idxs] = colors
        self.radii[update_idxs] = surfels.radii[update_idxs].view(-1, 1)

        self.draw_idx_set.add(update_idxs_cpu.numpy().tolist())

        # render_surfels.mark_invisible(surfel_removal.cpu())

    def mark_visible(self, update_indices):
        pass

    def unmark_visible(self, removal_indices):
        pass

    def update_indices(self):
        pass


def _test():
    import tenviz.io

    ctx = tenviz.Context()
    geo = tenviz.io.read_3dobject(
        Path(__file__).parent / "../../test-data/bunny/bun_zipper.ply").torch()

    with ctx.current():
        surfels = SurfelCloud(geo.verts.size(0))
        surfels.points.from_tensor(geo.verts)
        surfels.normals.from_tensor(geo.normals)
        colors = (torch.abs(torch.rand(geo.verts.size(0), 3))*255).byte()
        surfels.colors.from_tensor(colors)

        surfels.radii.from_tensor(torch.rand(
            geo.verts.size(0), 1, dtype=torch.float32).abs()*0.025)

    viewer = ctx.viewer([surfels], tenviz.CameraManipulator.WASD)

    while True:
        key = viewer.wait_key(1)
        if key == 27:
            break


if __name__ == '__main__':
    _test()
