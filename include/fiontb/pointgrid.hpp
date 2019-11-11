#pragma once

#include <torch/torch.h>

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace fiontb {
template <Device dev>
struct BasePointGrid {
  const typename Accessor<dev, bool, 2>::T mask;
  const int width;
  const int height;

  BasePointGrid(const torch::Tensor mask)
      : mask(Accessor<dev, bool, 2>::Get(mask)),
        width(mask.size(1)),
        height(mask.size(0)) {}

  FTB_DEVICE_HOST bool empty(int row, int col) const { return !mask[row][col]; }
};

}  // namespace fiontb
