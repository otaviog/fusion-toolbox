import torch

from fiontb._cfiontb import SurfelFusionOp as _SurfelFusionOp
from fiontb._utils import empty_ensured_size


class CarveSpace:
    """Remove unstable surfels in front of recently updated stable
    surfels.

    """

    def __init__(self, stable_conf_thresh,
                 search_size=2, min_z_difference=.5):
        self.stable_conf_thresh = stable_conf_thresh
        self.search_size = search_size
        self.min_z_difference = min_z_difference

        self._free_map = None

    def __call__(self, indexmap, current_time, model, update_gl=False):
        indexmap.synchronize()
        ref_device = indexmap.position_confidence.device
        self._free_map = empty_ensured_size(
            self._free_map, indexmap.height,
            indexmap.width,
            dtype=torch.int64,
            device=ref_device)

        _SurfelFusionOp.carve_space(indexmap, self._free_map,
                                    current_time, self.stable_conf_thresh,
                                    self.search_size, self.min_z_difference)

        deleted = self._free_map[self._free_map > -1]
        model.free(deleted, update_gl=update_gl)
        return deleted.size(0)
