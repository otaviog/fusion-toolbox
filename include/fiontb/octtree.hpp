#pragma once

#include <torch/torch.h>

#include "aabb.hpp"

namespace fiontb {
namespace priv {
class OctNode;
}

class OctTree {
 public:
  OctTree(torch::Tensor points, int leaf_num_points = 8);
  OctTree(const OctTree &) = delete;
  OctTree &operator=(OctTree const &) = delete;

  ~OctTree();

  std::pair<torch::Tensor, torch::Tensor> Query(const torch::Tensor &qpoints,
                                                int max_k, float radius);

 private:
  void QueryPoint(const Eigen::Vector3f &qpoint, int max_k);

  AABB box_;
  priv::OctNode *root_;
  torch::Tensor points_;
};
}  // namespace fiontb
