import torch

from fiontb._cfiontb import FSFOp
from fiontb._utils import empty_ensured_size


class MergeLive:
    def __init__(self, search_size, max_normal_angle):
        self.search_size = search_size
        self.max_normal_angle = max_normal_angle
        self._new_surfels_map = None

    def __call__(self, target_indexmap, live_indexmap, mapped_model):
        self._new_surfels_map = empty_ensured_size(
            self._new_surfels_map,
            live_indexmap.height,
            live_indexmap.width, dtype=torch.int64,
            device=target_indexmap.position_confidence.device)

        FSFOp.merge_live(target_indexmap,
                         live_indexmap,
                         mapped_model,
                         self.search_size,
                         self.max_normal_angle,
                         self._new_surfels_map)

        return self._new_surfels_map
