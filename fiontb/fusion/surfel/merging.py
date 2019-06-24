from pathlib import Path

import torch

from ._raster import GlobalSurfelRaster
from ._ckernels import surfel_merge_redundant


class MergingContext(GlobalSurfelRaster):
    def __init__(self, surfel_model):
        super(MergingContext, self).__init__(surfel_model)

    def merge_close_surfels(self, proj_matrix, rt_cam, width, height,
                            stable_conf_thresh):
        self.raster(proj_matrix, rt_cam, width, height,
                    stable_conf_thresh, -1)
        context = self.surfel_model.context

        with context.current():
            pos_fb = self.pos_tex.to_tensor()
            normal_rad_fb = self.normal_rad_tex.to_tensor()
            idx_fb = self.idx_tex.to_tensor()

        surfel_merge_redundant(pos_fb, normal_rad_fb,
                               idx_fb, self.surfel_model.active_mask,
                               0.05, 34, 4)


def _test():
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

    radii = torch.full((model_size, ), 0.05, dtype=torch.float)
    confs = torch.full((model_size, ), 15, dtype=torch.float)
    times = torch.full((model_size, ), 5, dtype=torch.int32)

    model_cloud = SurfelCloud(model.verts, model.colors, model.normals,
                              radii, confs, times, "cpu")
    model_cloud.to("cuda:0")

    surfel_model = SurfelModel(ctx, model_size*2)
    surfel_model.add_surfels(model_cloud)
    surfel_model.update_active_mask_gl()

    before = surfel_model.compact()
    
    merge_ctx = MergingContext(surfel_model)
    merge_ctx.merge_close_surfels(proj_matrix, rt_cam, 640, 480,
                                  15)
    surfel_model.update_active_mask_gl()

    show_surfels(ctx, [before, surfel_model])


if __name__ == "__main__":
    _test()
