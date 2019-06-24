#pragma once

#include <torch/torch.h>

#include "aabb.hpp"

namespace fiontb {
class ATrigOctNode;

class TrigOctree {
 public:
  TrigOctree(torch::Tensor verts, const torch::Tensor &faces,
             int leaf_num_trigs);
  ~TrigOctree();

  std::pair<torch::Tensor, torch::Tensor> QueryClosest(
      const torch::Tensor &points);

 private:
  ATrigOctNode *root_;
  torch::Tensor verts_;
};
}  // namespace fiontb
