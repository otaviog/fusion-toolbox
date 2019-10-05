import math

import torch

from fiontb._cfiontb import SurfelFusionOp as _SurfelFusionOp
from fiontb._utils import empty_ensured_size


class Merge:

    def __init__(self, max_distance=0.005, normal_max_angle=math.radians(30), search_size=4,
                 stable_conf_thresh=10):
        self.max_dist = max_distance
        self.normal_max_angle = normal_max_angle
        self.search_size = search_size
        self.stable_conf_thresh = stable_conf_thresh

        self._merge_map = None

    def __call__(self, model_indexmap, model, update_gl=False):
        ref_device = model_indexmap.position_confidence.device

        self._merge_map = empty_ensured_size(self._merge_map,
                                             model_indexmap.height,
                                             model_indexmap.width,
                                             dtype=torch.int64,
                                             device=ref_device)
        with model.gl_context.current():
            with model.map_as_tensors(ref_device) as mapped_model:
                _SurfelFusionOp.merge(model_indexmap, self._merge_map, mapped_model, self.max_dist,
                                      self.normal_max_angle, self.search_size,
                                      self.stable_conf_thresh)

            deleted = self._merge_map[self._merge_map > -1]
            model.free(deleted, update_gl)

            return deleted.size(0)
