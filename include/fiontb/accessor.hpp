#pragma once

#include <torch/torch.h>

namespace fiontb {
template <bool CUDA, typename scalar_t, unsigned long dims>
struct Accessor {
  typedef torch::TensorAccessor<scalar_t, dims> Type;

  static Type Get(const torch::Tensor &tensor) {
    return tensor.accessor<scalar_t, dims>();
  }
};

template <typename scalar_t, unsigned long dims>
using CPUAccessor = Accessor<false, scalar_t, dims>;

#ifdef __CUDACC__
template <>
struct Accessor<true> {
  typedef torch::TensorPackedAccessor<scalar_t, dims, torch::RestrictPtrTraits,
                                      size_t>
      Type;

  static Type Get(const torch::Tensor &tensor) {
    return tensor.packed_accessor<scalar_t, dims>();
  }
};
template <typename scalar_t, unsigned long dims>
using CUDAAccessor = Accessor<true, scalar_t, dims>;
#endif
}  // namespace fiontb
