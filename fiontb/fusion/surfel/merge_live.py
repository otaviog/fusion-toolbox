import math

import torch

from fiontb._cfiontb import SurfelFusionOp
from fiontb._utils import empty_ensured_size

from .indexmap import LiveIndexMapRaster


class MergeLiveSurfels:
    def __init__(self, gl_context, search_size=2, max_normal_angle=math.radians(30)):
        self.search_size = search_size
        self.max_normal_angle = max_normal_angle
        self.live_raster = LiveIndexMapRaster(gl_context)

        self._new_surfels_map = None
        self._gl_context = gl_context

    def __call__(self, target_indexmap, live_surfels, gl_proj_matrix,
                 rt_cam, width, height, surfel_model):
        ref_device = target_indexmap.position_confidence.device

        self._new_surfels_map = empty_ensured_size(
            self._new_surfels_map,
            height, width, dtype=torch.int64,
            device=ref_device)

        self.live_raster.raster(live_surfels, gl_proj_matrix, width, height)
        with self._gl_context.current():
            live_indexmap = self.live_raster.to_indexmap(ref_device)
            with surfel_model.map_as_tensors(ref_device) as mapped_model:
                SurfelFusionOp.merge_live(target_indexmap,
                                          live_indexmap,
                                          mapped_model,
                                          rt_cam.cam_to_world.to(ref_device),
                                          self.search_size,
                                          self.max_normal_angle,
                                          self._new_surfels_map)

        new_surfels_index = self._new_surfels_map[self._new_surfels_map > -1]
        new_surfels = live_surfels[new_surfels_index]

        new_surfels.itransform(rt_cam.cam_to_world)
        return new_surfels
