#pragma once

#include <torch/torch.h>

#include "accessor.hpp"

namespace fiontb {
template <bool CUDA>
class BasePointGrid {
 public:
  BasePointGrid(const torch::Tensor mask)
      : mask(Accessor<CUDA, uint8_t, 2>::Get(mask)) {}

  bool empty(int row, int col) const { return mask[row][col] == 0; }

  typename Accessor<CUDA, uint8_t, 2>::Type mask;
};

}  // namespace fiontb
