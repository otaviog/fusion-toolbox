class RemoveUnstable:
    def __init__(self, stable_conf_thresh=10, max_unstable_time=20):
        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

    def __call__(self, indexmap, time, model, update_gl=False):
        visible_idxs = indexmap[:, :, 0][indexmap[:, :, 1] == 1].long()

        if visible_idxs.size(0) == 0:
            return 0

        with model.gl_context.current():
            confs = model.confidences[visible_idxs].squeeze()
            times = model.times[visible_idxs].squeeze()

        model.max_confidence = max(
            confs.max().item(), model.max_confidence)

        unstable_idxs = visible_idxs[(confs < self.stable_conf_thresh)
                                     & (time - times >= self.max_unstable_time)]

        if unstable_idxs.size(0) == 0:
            return 0

        model.free(unstable_idxs, update_gl)
        return unstable_idxs.size(0)
