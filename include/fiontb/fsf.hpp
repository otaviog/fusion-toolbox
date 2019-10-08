#pragma once

#include <string>

#include <torch/torch.h>

#include "surfel_fusion.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {

struct FSFOp {
  static void RegisterPybind(pybind11::module &m);

  static void Merge(const torch::Tensor &knn_index,
                    const SurfelCloud &local_model,
                    const torch::Tensor &global_map,
                    MappedSurfelModel global_model);
};
}  // namespace fiontb
