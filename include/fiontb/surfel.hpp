#pragma once

#include <string>
#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
class SurfelAllocator {
 public:
  SurfelAllocator(int max_surfels, const std::string &device);

  static void RegisterPybind(pybind11::module &m);

  void FindFree(torch::Tensor out_free_indices);

  torch::Tensor free_mask_byte, free_mask_bit;
};

}  // namespace fiontb
