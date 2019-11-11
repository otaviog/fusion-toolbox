import math

import torch

from fiontb._cfiontb import SurfelFusionOp, ElasticFusionOp
from fiontb._utils import empty_ensured_size

_INT_MAX = 2147483647


class Update:
    def __init__(self, elastic_fusion=False,
                 search_size=2, max_normal_angle=math.radians(30)):
        self.elastic_fusion = elastic_fusion
        self.search_size = search_size
        self.max_normal_angle = max_normal_angle
        self._new_surfels_map = None

    def __call__(self, model_indexmap, live_surfels, kcam,
                 rt_cam, time, surfel_model):
        ref_device = model_indexmap.position_confidence.device

        self._new_surfels_map = empty_ensured_size(
            self._new_surfels_map,
            live_surfels.size, dtype=torch.bool,
            device=ref_device)

        model_merge_map = torch.full((model_indexmap.height, model_indexmap.width, 3),
                                     _INT_MAX, dtype=torch.int32, device=ref_device)
        with surfel_model.gl_context.current():
            with surfel_model.map_as_tensors(ref_device) as mapped_model:

                scale = int(model_indexmap.height / kcam.image_height)

                if self.elastic_fusion:
                    ElasticFusionOp.update(
                        model_indexmap, live_surfels.to_cpp_(), mapped_model,
                        kcam.matrix.to(ref_device), rt_cam.cam_to_world,
                        self.search_size, time, scale,
                        model_merge_map, self._new_surfels_map)
                else:
                    SurfelFusionOp.update(
                        model_indexmap, live_surfels.to_cpp_(), mapped_model,
                        kcam.matrix.to(ref_device), rt_cam.cam_to_world,
                        self.max_normal_angle, self.search_size, time, scale,
                        model_merge_map, self._new_surfels_map)

        new_surfels = live_surfels[self._new_surfels_map]

        new_surfels.itransform(rt_cam.cam_to_world)
        return new_surfels
