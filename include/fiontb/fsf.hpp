#pragma once

#include <string>

#include <torch/torch.h>

#include "surfel_fusion.hpp"

namespace fiontb {

struct FSFOp {
  static void MergeLive(const IndexMap &target_indexmap,
                        const IndexMap &live_indexmap,
                        const MappedSurfelModel &model,
                        const torch::Tensor &rt_cam, int search_size,
                        float max_normal_angle, torch::Tensor new_surfels_map);
};
}  // namespace fiontb
