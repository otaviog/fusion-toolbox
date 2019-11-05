import torch

from fiontb._cfiontb import SurfelFusionOp as _SurfelFusionOp
from fiontb._utils import empty_ensured_size


class CarveSpace:
    """Remove unstable surfels in front of recently updated stable
    surfels.

    """

    def __init__(self, stable_conf_thresh=10, stable_time_thresh=20,
                 search_size=2, min_z_difference=5e-2):
        self.stable_conf_thresh = stable_conf_thresh
        self.stable_time_thresh = stable_time_thresh
        self.search_size = search_size
        self.min_z_difference = min_z_difference

        self._free_map = None

    def __call__(self, kcam, rt_cam, indexmap, time, model, update_gl=False):
        ref_device = kcam.device
        alloc_indices = model.allocated_indices().to(ref_device)

        if alloc_indices.size(0) == 0:
            return 0

        remove_mask = torch.zeros(alloc_indices.size(0), device=ref_device,
                                  dtype=torch.bool)

        with model.gl_context.current():
            with model.map_as_tensors(ref_device) as mapped_model:
                _SurfelFusionOp.carve_space(mapped_model, alloc_indices,
                                            indexmap, kcam.matrix,
                                            rt_cam.world_to_cam,
                                            time, 2, self.stable_conf_thresh,
                                            self.stable_time_thresh,
                                            remove_mask)

        remove_idxs = alloc_indices[remove_mask]
        if remove_idxs.size(0) == 0:
            return 0

        model.free(remove_idxs, update_gl)
        return remove_idxs.size(0)
