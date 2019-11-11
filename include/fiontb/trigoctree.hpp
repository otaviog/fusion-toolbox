#pragma once

#include <torch/torch.h>

#include "aabb.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {
class ATrigOctNode;

class TrigOctree {
 public:
  TrigOctree(torch::Tensor verts, const torch::Tensor &faces,
             int leaf_num_trigs);
  ~TrigOctree();

  std::pair<torch::Tensor, torch::Tensor> QueryClosest(
      const torch::Tensor &points);

  static void RegisterPybind(pybind11::module &m);
  
 private:
  ATrigOctNode *root_;
  torch::Tensor verts_;
};
}  // namespace fiontb
