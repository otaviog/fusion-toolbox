import torch

from fiontb._cfiontb import SurfelFusionOp as _SurfelFusionOp


class Clean1:
    def __init__(self, stable_conf_thresh=10, max_unstable_time=20):
        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

    def __call__(self, indexmap, time, model, update_gl=False):
        alloc_indices = model.allocated_indices().to(model.device)

        if alloc_indices.size(0) == 0:
            return 0

        with model.gl_context.current():
            confs = model.confidences[alloc_indices].squeeze()
            times = model.times[alloc_indices].squeeze()

        model.max_confidence = max(
            confs.max().item(), model.max_confidence)

        unstable_idxs = alloc_indices[(confs < self.stable_conf_thresh)
                                      & (time - times >= self.max_unstable_time)]

        if unstable_idxs.size(0) == 0:
            return 0

        model.free(unstable_idxs, update_gl)
        return unstable_idxs.size(0)


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
                _SurfelFusionOp.clean(mapped_model, alloc_indices,
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
