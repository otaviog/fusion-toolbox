"""Routine for adding new surfels to a model.
"""
import math

import torch

from slamtb._cslamtb import SurfelFusionOp, ElasticFusionOp
from slamtb._utils import empty_ensured_size

from ._merge_map import create_merge_map


class Update:
    def __init__(self, elastic_fusion=False,
                 search_size=2, max_normal_angle=math.radians(30)):
        self.elastic_fusion = elastic_fusion
        self.search_size = search_size
        self.max_normal_angle = max_normal_angle
        self._new_surfels_map = None

    def __call__(self, model_indexmap, live_surfels, kcam,
                 rt_cam, time, surfel_model):
        ref_device = model_indexmap.point_confidence.device

        self._new_surfels_map = empty_ensured_size(
            self._new_surfels_map,
            live_surfels.size, dtype=torch.bool,
            device=ref_device)
        model_merge_map = create_merge_map(
            model_indexmap.width, model_indexmap.height, ref_device)
        with surfel_model.gl_context.current():
            with surfel_model.map_as_tensors(ref_device) as mapped_model:

                scale = int(model_indexmap.height / kcam.image_height)

                if self.elastic_fusion:
                    ElasticFusionOp.update(
                        model_indexmap, live_surfels.to_cpp_(), mapped_model,
                        kcam.matrix.to(ref_device),
                        rt_cam.cam_to_world.float(),
                        self.search_size, time, scale,
                        model_merge_map, sel._new_surfels_map)
                else:
                    merge_corresp = torch.full((model_indexmap.width * model_indexmap.height, 2),
                                               -1, device=ref_device, dtype=torch.int64)

                    SurfelFusionOp.find_updatable(model_indexmap,
                                                  live_surfels.to_cpp_(),
                                                  kcam.matrix.to(ref_device),
                                                  self.max_normal_angle, self.search_size, time,
                                                  scale, model_merge_map, self._new_surfels_map,
                                                  merge_corresp)

                    merge_corresp = merge_corresp[merge_corresp[:, 0] > -1, :]
                    SurfelFusionOp.update(merge_corresp,
                                          live_surfels.to_cpp_(),
                                          mapped_model,
                                          rt_cam.cam_to_world.float(), time)
                    if live_surfels.sparse_features is not None:
                        surfel_model.sparse_features.merge(
                            merge_corresp.cpu(), live_surfels.sparse_features)
        new_surfels = live_surfels[self._new_surfels_map.nonzero().squeeze()]
        new_surfels.itransform(rt_cam.cam_to_world)
        return new_surfels
