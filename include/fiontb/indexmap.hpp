#pragma once

#include <torch/torch.h>

namespace fiontb {
class IndexMap {
 public:
  IndexMap(torch::Tensor proj_points, torch::Tensor model_points,
           int width, int height, int window_size, int depth_slots);

  std::pair<torch::Tensor, torch::Tensor> Query(
      const torch::Tensor &proj_query_points, const torch::Tensor &query_points,
      int query_k);

  torch::Tensor grid_, proj_points_, model_points_;
  int depth_slots_, window_size_;
};
}  // namespace fiontb
