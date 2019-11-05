import torch

from fiontb._cfiontb import ElasticFusionOp


class Clean:
    def __init__(self, stable_conf_thresh=10, max_unstable_time=20):
        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

    def __call__(self, kcam, rt_cam, indexmap, time, model, update_gl=False):
        ref_device = kcam.device
        alloc_indices = model.allocated_indices().to(ref_device)

        if alloc_indices.size(0) == 0:
            return 0

        remove_mask = torch.zeros(alloc_indices.size(0), device=ref_device,
                                  dtype=torch.bool)

        with model.gl_context.current():
            with model.map_as_tensors(ref_device) as mapped_model:
                ElasticFusionOp.clean(mapped_model, alloc_indices,
                                      indexmap, kcam.matrix,
                                      rt_cam.world_to_cam.to(ref_device),
                                      time, self.max_unstable_time,
                                      2, self.stable_conf_thresh,
                                      remove_mask)

        remove_idxs = alloc_indices[remove_mask]
        if remove_idxs.size(0) == 0:
            return 0

        model.free(remove_idxs, update_gl)
        return remove_idxs.size(0)
