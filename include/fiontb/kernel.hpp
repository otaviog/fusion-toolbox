#pragma once

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace fiontb {
#ifdef __CUDACC__
template <typename Kernel>
static __global__ void Exec1DKernel(Kernel kern, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    kern(idx);
  }
}
#endif

template <Device dev, typename Kernel>
inline void Launch1DKernel(Kernel kern, int size) {
  if (dev == kCUDA) {
#ifdef __CUDACC__
    CudaKernelDims kl = Get1DKernelDims(size);
    Exec1DKernel<<<kl.grid, kl.block>>>(kern, size);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
#endif
  } else {
    for (int i = 0; i < size; ++i) {
      kern(i);
    }
  }
}

}  // namespace fiontb
