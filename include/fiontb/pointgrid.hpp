#pragma once

#include <torch/torch.h>

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace fiontb {
template <Device dev>
class BasePointGrid {
 public:
  BasePointGrid(const torch::Tensor mask)
      : mask(Accessor<dev, uint8_t, 2>::Get(mask)) {}

  FTB_DEVICE_HOST bool empty(int row, int col) const {
    return mask[row][col] == 0;
  }

  const typename Accessor<dev, uint8_t, 2>::T mask;
};

}  // namespace fiontb
