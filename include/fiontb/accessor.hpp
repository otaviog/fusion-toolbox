#pragma once

#include <torch/torch.h>

namespace fiontb {

enum Device { kCPU = 0, kCUDA = 1 };

template <Device dev, typename scalar_t, unsigned long dims>
struct Accessor {
  typedef torch::TensorAccessor<scalar_t, dims> T;

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

  static T Get(torch::Tensor &tensor) {
    return tensor.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>();
  }

  static T Get(const torch::Tensor &tensor) {
    return tensor.packed_accessor<scalar_t, dims, torch::RestrictPtrTraits, size_t>();
  }
};
template <typename scalar_t, unsigned long dims>
using CUDAAccessor = Accessor<kCUDA, scalar_t, dims>;
#endif
}  // namespace fiontb
