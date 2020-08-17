import math

import torch

from slamtb._cslamtb import SurfelFusionOp as _SurfelFusionOp
from slamtb._utils import empty_ensured_size


class Merge:

    def __init__(self, max_distance=0.01, normal_max_angle=math.radians(30),
                 search_size=4, stable_conf_thresh=10):
        self.max_dist = max_distance
        self.normal_max_angle = normal_max_angle
        self.search_size = search_size
        self.stable_conf_thresh = stable_conf_thresh

        self._merge_map = None

    def __call__(self, model_indexmap, model, update_gl=False):
        ref_device = model_indexmap.point_confidence.device

        self._merge_map = empty_ensured_size(self._merge_map,
                                             model_indexmap.height,
                                             model_indexmap.width,
                                             dtype=torch.int64,
                                             device=ref_device)
        with model.gl_context.current():
            model_indexmap.synchronize()
            with model.map_as_tensors(ref_device) as mapped_model:
                merge_corresp = _SurfelFusionOp.find_mergeable(
                    model_indexmap, self._merge_map, self.max_dist,
                    self.normal_max_angle, self.search_size,
                    self.stable_conf_thresh)

                merge_corresp = merge_corresp[merge_corresp[:, 0] > -1, :]
                _SurfelFusionOp.merge(merge_corresp, mapped_model)

            model.sparse_features.merge(merge_corresp.cpu())
            model.free(merge_corresp[:, 1], update_gl)

            return merge_corresp.size(0)
