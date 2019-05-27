#pragma once

#include <torch/torch.h>

namespace fiontb {
class IndexMap {
 public:
  IndexMap(torch::Tensor proj_points, int map_width, int map_height);

  void Pairwise();

  std::pair<torch::Tensor, torch::Tensor> Query(
      const torch::Tensor &query_points, int query_kj);

  torch::Tensor grid_, model_points_;
};
}  // namespace fiontb
