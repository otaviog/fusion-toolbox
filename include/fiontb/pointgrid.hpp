#pragma once

#include <torch/torch.h>

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace fiontb {
template <Device dev>
class BasePointGrid {
 public:
  BasePointGrid(const torch::Tensor mask)
      : mask(Accessor<dev, bool, 2>::Get(mask)) {}

  FTB_DEVICE_HOST bool empty(int row, int col) const {
    return !mask[row][col];
  }

  const typename Accessor<dev, bool, 2>::T mask;
};

}  // namespace fiontb
