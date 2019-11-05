#pragma once

#include <torch/torch.h>

#include "surfel_fusion.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {

struct ElasticFusionOp {
  static void Update(const IndexMap &target_indexmap,
                     const IndexMap &live_indexmap,
                     const torch::Tensor &live_features,
                     MappedSurfelModel model, const torch::Tensor &rt_cam,
                     int search_size, int time,
                     torch::Tensor model_merge_map,
                     torch::Tensor new_surfels_map);

  static void Clean(MappedSurfelModel model, torch::Tensor model_indices,
                    const IndexMap &model_indexmap, const torch::Tensor &kcam,
                    const torch::Tensor &world_to_cam, int time,
                    int max_time_thresh, int neighbor_size,
                    float stable_conf_thresh, torch::Tensor remove_mask);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
