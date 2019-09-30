#pragma once

#include <torch/torch.h>

#include "device.hpp"

namespace fiontb {

template <Device dev, typename scalar_t, unsigned long dims>
struct Accessor {
  typedef torch::TensorAccessor<scalar_t, dims> T;
  typedef torch::TensorAccessor<scalar_t, dims> Ts;

  static T Get(torch::Tensor &tensor) {
    return tensor.accessor<scalar_t, dims>();
  }

  static T Get(const torch::Tensor &tensor) {
    return tensor.accessor<scalar_t, dims>();
  }
};

template <typename scalar_t, unsigned long dims>
using CPUAccessor = Accessor<kCPU, scalar_t, dims>;

#ifdef __CUDACC__
template <typename scalar_t, unsigned long dims>
struct Accessor<kCUDA, scalar_t, dims> {
  typedef torch::PackedTensorAccessor<scalar_t, dims, torch::RestrictPtrTraits,
                                      size_t>
      T;

  typedef torch::TensorAccessor<scalar_t, dims, torch::RestrictPtrTraits,
                                      size_t>
  Ts;

  static T Get(torch::Tensor &tensor) {
    return tensor
        .packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>();
  }

  static T Get(const torch::Tensor &tensor) {
    return tensor
        .packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>();
  }
};
template <typename scalar_t, unsigned long dims>
using CUDAAccessor = Accessor<kCUDA, scalar_t, dims>;
#endif
}  // namespace fiontb
