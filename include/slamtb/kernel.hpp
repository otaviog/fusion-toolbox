#pragma once

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace slamtb {
#ifdef __CUDACC__

/**
 * Used by Launch1DKernelCUDA
 *
 */
template <typename Kernel>
static __global__ void Exec1DKernel(Kernel kern, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    kern(idx);
  }
}

/**
 * Launches a kernel functor on CUDA. The launch configuration is
 * determined automatically by the given size.
 *
 * After execution errors are verified and the device is synchronized.
 *
 * @param kern Kernel functor.
 * @param size the number of elements to process.
 */
template <typename Kernel>
inline void Launch1DKernelCUDA(Kernel &kern, int size) {
  CudaKernelDims kl = Get1DKernelDims(size);
  Exec1DKernel<<<kl.grid, kl.block>>>(kern, size);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename Kernel>
static __global__ void Exec2DKernel(Kernel kern, int width, int height) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    kern(row, col);
  }
}

template <typename Kernel>
inline void Launch2DKernelCUDA(Kernel &kern, int width, int height) {
  CudaKernelDims kl = Get2DKernelDims(width, height);
  Exec2DKernel<<<kl.grid, kl.block>>>(kern, width, height);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

#endif

template <typename Kernel>
inline void Launch1DKernelCPU(Kernel &kern, int size, bool sequential = false) {
  if (!sequential) {
#ifdef NDEBUG
#pragma omp parallel for
#endif
    for (int i = 0; i < size; ++i) {
      kern(i);
    }
  } else {
    for (int i = 0; i < size; ++i) {
      kern(i);
    }
  }
}

template <typename Kernel>
inline void Launch2DKernelCPU(Kernel &kern, int width, int height) {
#ifdef NDEBUG
#pragma omp parallel for
#endif
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      kern(row, col);
    }
  }
}

}  // namespace slamtb
