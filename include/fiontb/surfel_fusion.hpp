#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor FilterSearch(torch::Tensor dist_mtx, torch::Tensor idx_mtx,
                           torch::Tensor live_normals,
                           torch::Tensor model_normals, float min_dist,
                           float min_normal_dot);

class IndexMap {
 public:
  IndexMap(int width, int height,
           torch::Tensor proj_points);

  void Pairwise();
  
  std::pair<torch::Tensor, torch::Tensor> Query(const torch::Tensor &query_points, int query_kj);
  
  torch::Tensor grid_, model_points_;
};
}
