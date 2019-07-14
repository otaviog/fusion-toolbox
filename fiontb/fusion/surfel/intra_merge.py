"""Surfel merging. More info on: Keller,
Maik, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim Weyrich, and
Andreas Kolb. "Real-time 3d reconstruction in dynamic scenes using
point-based fusion." In 2013 International Conference on 3D Vision-3DV
2013, pp. 1-8. IEEE, 2013.
"""

from pathlib import Path
import math

import torch

from .indexmap import ModelIndexMap
from ._ckernels import surfel_find_mergeable_surfels


class IntraMergeMap(ModelIndexMap):
    """Finds and merge surfel that are too close.
    """

    def __init__(self, surfel_model, max_dist=0.005,
                 normal_max_angle=math.radians(45),
                 search_size=2):
        super(IntraMergeMap, self).__init__(surfel_model)
        self.merge_map = None
        self.max_dist = max_dist
        self.normal_max_angle = normal_max_angle
        self.search_size = search_size

    def find_mergeable_surfels(self, proj_matrix, rt_cam, width, height,
                               stable_conf_thresh):
        self.raster(proj_matrix, rt_cam, width, height,
                    stable_conf_thresh, -1)
        context = self.surfel_model.context

        with context.current():
            pos_fb = self.pos_tex.to_tensor()
            normal_rad_fb = self.normal_rad_tex.to_tensor()
            idx_fb = self.idx_tex.to_tensor()

        if self.merge_map is None:
            self.merge_map = torch.empty(pos_fb.size(0), pos_fb.size(1),
                                         device=pos_fb.device, dtype=torch.int64)

        surfel_find_mergeable_surfels(
            pos_fb, normal_rad_fb,
            idx_fb, self.merge_map,
            self.max_dist, self.normal_max_angle, self.search_size)

        which = self.merge_map > -1
        dest_idxs = self.merge_map[which]

        merge_idxs = idx_fb[:, :, 0][which].long()

        return dest_idxs, merge_idxs


def _test():
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.viz.surfelrender import show_surfels
    from .model import SurfelModel, SurfelCloud

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

    radii = torch.full((model_size, ), 0.005, dtype=torch.float)
    confs = torch.full((model_size, ), 15, dtype=torch.float)
    times = torch.full((model_size, ), 5, dtype=torch.int32)

    model_cloud = SurfelCloud(model.verts, model.colors, model.normals,
                              radii, confs, times, None, "cpu")
    model_cloud.to("cuda:0")

    surfel_model = SurfelModel(ctx, model_size*2)
    surfel_model.add_surfels(model_cloud)
    surfel_model.update_active_mask_gl()

    before = surfel_model.compact()

    merge_ctx = IntraMergeMap(surfel_model)
    dest_idxs, merge_idxs = merge_ctx.find_mergeable_surfels(proj_matrix, rt_cam, 640, 480,
                                                             15)
    surfel_model.mark_inactive(merge_idxs)
    surfel_model.update_active_mask_gl()

    show_surfels(ctx, [before, surfel_model])


if __name__ == "__main__":
    _test()
