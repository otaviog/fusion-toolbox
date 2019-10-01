#pragma once

#include <torch/torch.h>

namespace fiontb {
class SurfelAllocator {
 public:
  SurfelAllocator(int max_surfels, const std::string &device);

  void FindUnactive(torch::Tensor unactive_indices);

  torch::Tensor free_mask_byte, free_mask_bit;
};

}  // namespace fiontb
