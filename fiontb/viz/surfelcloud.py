
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import tenviz


_SHADER_DIR = Path(__file__).parent / 'shaders'


class SurfelCloud(tenviz.DrawProgram):
    def __init__(self, num_surfels):
        self.points = tenviz.buffer_empty(num_surfels, 3, tenviz.BType.Float)
        self.normals = tenviz.buffer_empty(num_surfels, 3, tenviz.BType.Float)
        self.colors = tenviz.buffer_empty(num_surfels, 3, tenviz.BType.Uint8)
        self.radii = tenviz.buffer_empty(num_surfels, 1, tenviz.BType.Float)
        self.counters = tenviz.buffer_empty(num_surfels, 1, tenviz.BType.Float)

        super(SurfelCloud, self).__init__(tenviz.DrawMode.Points,
                                          vert_shader=_SHADER_DIR / "surfel.vert",
                                          frag_shader=_SHADER_DIR / "surfel.frag",
                                          geo_shader=_SHADER_DIR / "surfel.geo",
                                          ignore_missing=False)

        self['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        self['in_pos'] = self.points
        self['in_color'] = self.colors
        self['in_normal'] = self.normals
        self['in_radius'] = self.radii
        # self['in_count'] = self.counters

        surfel_tex = np.array(Image.open(str(Path(__file__).parent
                                             / "assets/surfel.png")))
        surfel_tex = torch.from_numpy(surfel_tex)

        self['BaseTexture'] = tenviz.tex_from_torch(surfel_tex)
        self.draw_idx_set = set()
        self.indices.from_tensor(torch.zeros(num_surfels, dtype=torch.int32))

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
        self.draw_idx_set.remove(removal_idxs.numpy().tolist())
        indices = torch.tensor(self.draw_idx_set, torch.int64)
        self.indices.copy_tensor(indices)
        
        # render_surfels.mark_invisible(surfel_removal.cpu())

    def mark_visible(self, update_indices):
        pass

    def unmark_visible(self, removal_indices):
        pass

    #def update_indices(self, update_indices, removal_indices):
        #self.


def _test():
    import tenviz.io
    from fiontb.fusion.surfel import SceneSurfelData

    ctx = tenviz.Context()
    geo = tenviz.io.read_3dobject(
        Path(__file__).parent / "../../test-data/bunny/bun_zipper_res4.ply").torch()

    normals = tenviz.geometry.compute_normals(geo.verts, geo.faces)
    device = "cuda:0"
    surfels = SceneSurfelData(geo.verts.size(0), device)

    surfels.points[:] = geo.verts
    surfels.normals[:] = normals

    colors = (torch.abs(torch.rand(geo.verts.size(0), 3))*255).byte()
    surfels.colors[:] = colors

    surfels.radii[:] = torch.rand(
        geo.verts.size(0), 1, dtype=torch.float32).abs()*0.01

    visible_indices = torch.arange(
        0, geo.verts.size(0), dtype=torch.int64)

    with ctx.current():
        surfel_render = SurfelCloud(geo.verts.size(0))

        surfel_render.update(surfels,  visible_indices,
                             torch.tensor([], dtype=torch.int64))

    viewer = ctx.viewer([surfels], tenviz.CameraManipulator.WASD)

    while True:
        key = viewer.wait_key(1)
        if key == 27:
            break


if __name__ == '__main__':
    _test()
