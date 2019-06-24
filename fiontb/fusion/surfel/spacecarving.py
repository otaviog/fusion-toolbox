from pathlib import Path

import torch

from ._raster import GlobalSurfelRaster
from ._ckernels import surfel_cave_free_space


class SpaceCarvingContext(GlobalSurfelRaster):
    def __init__(self, surfel_model):
        super(SpaceCarvingContext, self).__init__(surfel_model)

    def carve(self, proj_matrix, rt_cam, width, height,
              stable_conf_thresh, time, window_size, debug=False):
        self.raster(proj_matrix, rt_cam, width, height,
                    stable_conf_thresh, time)
        context = self.surfel_model.context

        with context.current():
            stable_positions = self.pos_tex.to_tensor()
            stable_normals = self.normal_rad_tex.to_tensor()
            stable_idxs = self.idx_tex.to_tensor()

        self.show_debug("Stable", debug)
        self.raster(proj_matrix, rt_cam, width, height,
                    -1.0, -1)

        with context.current():
            view_positions = self.pos_tex.to_tensor()
            view_normals = self.normal_rad_tex.to_tensor()
            view_idxs = self.idx_tex.to_tensor()

        self.show_debug("All", debug)
        surfel_cave_free_space(stable_positions, stable_idxs,
                               view_positions, view_idxs,
                               self.surfel_model.active_mask, window_size)


def _test():
    import numpy as np
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud
    from fiontb.viz.surfelrender import show_surfels

    test_data = Path(__file__).parent / "_test"

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))
    proj = tenviz.projection_from_kcam(
        kcam.matrix, 0.01, 10.0)
    proj_matrix = torch.from_numpy(proj.to_matrix()).float()

    rt_cam = RTCamera.create_from_pos_quat(
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681)

    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    radii = torch.full((model_size, ), 0.05, dtype=torch.float)
    confs = torch.full((model_size, ), 15, dtype=torch.float)
    times = torch.full((model_size, ), 5, dtype=torch.int32)

    model_cloud = SurfelCloud(model.verts, model.colors, model.normals,
                              radii, confs, times, "cpu")
    model_cloud.to("cuda:0")

    surfel_model.add_surfels(model_cloud)

    np.random.seed(10)
    space_vl_sample = np.random.choice(model.verts.size(0), 100)
    verts = (model.verts[space_vl_sample]
             + (torch.from_numpy(rt_cam.center()).float() -
                model.verts[space_vl_sample])
             * torch.rand(100, 1))
    space_vl_cloud = SurfelCloud(verts,
                                 model_cloud.colors[space_vl_sample],
                                 model_cloud.normals[space_vl_sample],
                                 radii[space_vl_sample],
                                 confs[space_vl_sample],
                                 torch.full((100, ), 3, dtype=torch.int32),
                                 "cpu")
    space_vl_cloud.to("cuda:0")
    surfel_model.add_surfels(space_vl_cloud)
    surfel_model.update_active_mask_gl()
    before = surfel_model.clone()

    carving_ctx = SpaceCarvingContext(surfel_model)
    carving_ctx.carve(proj_matrix, rt_cam, 640, 480, 10, 5, 4)
    surfel_model.update_active_mask_gl()

    show_surfels(ctx, [before, surfel_model])


if __name__ == "__main__":
    _test()
