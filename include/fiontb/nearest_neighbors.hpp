#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
struct NearestNeighborsOp {
  static void Forward(const torch::Tensor &nn_distances,
                      const torch::Tensor &nn_index,
                      const torch::Tensor &features,
                      torch::Tensor out_features);

  static void ComputeEpsilonDistances(const torch::Tensor &target_xyz,
                                      const torch::Tensor &source_xyz,
                                      const torch::Tensor &nn_index,
                                      const torch::Tensor &epsilon_distances);

  static void Backward(const torch::Tensor &epsilon_distances,
                       const torch::Tensor &nn_index,
                       const torch::Tensor &features,
                       const torch::Tensor &dl_features, torch::Tensor dl_xyz);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
