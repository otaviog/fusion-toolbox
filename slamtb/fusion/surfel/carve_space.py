import torch

from slamtb._cslamtb import SurfelFusionOp as _SurfelFusionOp
from slamtb._utils import empty_ensured_size


class CarveSpace:
    """Remove unstable surfels in front of recently updated stable
    surfels.

    """

    def __init__(self, stable_conf_thresh=10, stable_time_thresh=20,
                 search_size=2, z_tollerance=5e-2):
        self.stable_conf_thresh = stable_conf_thresh
        self.stable_time_thresh = stable_time_thresh
        self.search_size = search_size
        self.z_tollerance = z_tollerance

        self._free_map = None

    def __call__(self, kcam, rt_cam, indexmap, time, model, update_gl=False):
        ref_device = indexmap.point_confidence.device
        alloc_indices = model.allocated_indices().to(ref_device)

        if alloc_indices.size(0) == 0:
            return 0

        indexmap_remove_mask = torch.zeros(indexmap.height, indexmap.width, device=ref_device,
                                           dtype=torch.bool)

        with model.gl_context.current():
            with model.map_as_tensors(ref_device) as mapped_model:
                _SurfelFusionOp.carve_space(mapped_model, alloc_indices,
                                            indexmap, kcam.matrix.to(
                                                ref_device),
                                            rt_cam.world_to_cam.float().to(ref_device),
                                            time, 2, self.stable_conf_thresh,
                                            self.stable_time_thresh, self.z_tollerance,
                                            indexmap_remove_mask)

        deleted = indexmap.indexmap[:, :, 0][indexmap_remove_mask].long()
        model.free(deleted, update_gl)
        return deleted.size(0)
