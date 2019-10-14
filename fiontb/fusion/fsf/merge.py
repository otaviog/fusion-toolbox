from fiontb._cfiontb import FSFOp as _FSFOp


class Merge:

    def __call__(self, knn_index, local_model, global_map, model, update_gl=False):
        ref_device = knn_index.device

        with model.gl_context.current():
            with model.map_as_tensors(ref_device) as mapped_model:
                _FSFOp.merge(knn_index, local_model.to_cpp_(),
                             global_map.to(ref_device), mapped_model)

            new_mask = knn_index[:, 0] >= global_map.size(0)
            model.add_surfels(local_model[new_mask], update_gl=update_gl)
