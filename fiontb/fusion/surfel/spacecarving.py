"""Surfel space carving for removing outliers. More info on: Keller,
Maik, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim Weyrich, and
Andreas Kolb. "Real-time 3d reconstruction in dynamic scenes using
point-based fusion." In 2013 International Conference on 3D Vision-3DV
2013, pp. 1-8. IEEE, 2013.
"""

from pathlib import Path

import torch

from .indexmap import ModelIndexMap
from ._ckernels import surfel_cave_free_space


class SpaceCarving:
    """Remove unstable surfels in front of recently updated stable
    surfels.

    """

    def __init__(self, surfel_model, stable_conf_thresh,
                 search_size=2, min_z_difference=0.1):
        """
        Args:
            surfel_model
             (:obj:`fiontb.fusion.surfel.model.SurfelModel`): Surfel model.
        """

        self.surfel_model = surfel_model
        self.stable_and_new_indexmap = ModelIndexMap(surfel_model)
        self.stable_conf_thresh = stable_conf_thresh
        self.search_size = search_size
        self.min_z_difference = min_z_difference

    def carve(self, model_indexmap, proj_matrix, rt_cam, width, height,
              curr_time, debug=False):
        """
        Args:

            proj_matrix (:obj:`torch.Tensor`):

        """

        model_indexmap.show_debug("Model", debug)

        context = self.surfel_model.context
        with context.current():
            model_positions = model_indexmap.position_confidence_tex.to_tensor()
            model_idxs = model_indexmap.index_tex.to_tensor()

        self.stable_and_new_indexmap.raster(proj_matrix, rt_cam, width, height,
                                            self.stable_conf_thresh, curr_time)
        self.stable_and_new_indexmap.show_debug("Stable and new", debug)

        with context.current():
            stable_and_new_positions = (self.stable_and_new_indexmap.
                                        position_confidence_tex.to_tensor())
            stable_and_new_idxs = self.stable_and_new_indexmap.index_tex.to_tensor()

        surfel_cave_free_space(
            stable_and_new_positions.cpu(), stable_and_new_idxs.cpu(),
            model_positions.cpu(), model_idxs.cpu(),
            self.surfel_model.active_mask, self.search_size,
            self.min_z_difference)


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

    proj = tenviz.projection_from_kcam(
        kcam.scaled(4).matrix, 0.001, 10.0)
    proj_matrix4 = torch.from_numpy(proj.to_matrix()).float()

    rt_cam = RTCamera.create_from_pos_quat(
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681)

    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    radii = torch.full((model_size, ), 0.05, dtype=torch.float)
    confs = torch.full((model_size, ), 15, dtype=torch.float)
    times = torch.full((model_size, ), 5, dtype=torch.int32)
    stable_and_new = SurfelCloud(model.verts, model.colors, model.normals,
                                 radii, confs, times, None, "cuda:0")

    surfel_model.add_surfels(stable_and_new)

    np.random.seed(110)
    num_violations = 100
    violations_sampling = np.random.choice(model.verts.size(0), num_violations)
    violation_points = (model.verts[violations_sampling]
                        + (rt_cam.center - model.verts[violations_sampling])
                        * torch.rand(num_violations, 1)*0.8 - 0.1)
    violations = SurfelCloud(violation_points,
                             stable_and_new.colors[violations_sampling],
                             stable_and_new.normals[violations_sampling],
                             radii[violations_sampling],
                             confs[violations_sampling] + 10,
                             torch.full((num_violations, ), 3, dtype=torch.int32),
                             None, "cuda:0")

    surfel_model.add_surfels(violations)
    surfel_model.update_active_mask_gl()
    before = surfel_model.clone()
    print(violations_sampling)
    model_indexmap = ModelIndexMap(surfel_model)
    model_indexmap.raster(proj_matrix, rt_cam, 640*4, 480*4)

    carving = SpaceCarving(surfel_model, stable_conf_thresh=10,
                           search_size=8, min_z_difference=0.1)
    carving.carve(model_indexmap, proj_matrix, rt_cam,
                  640*4, 480*4, 5, debug=False)
    surfel_model.update_active_mask_gl()

    show_surfels(ctx, [before, surfel_model],
                 view_matrix=rt_cam.opengl_view_cam)


if __name__ == "__main__":
    _test()
