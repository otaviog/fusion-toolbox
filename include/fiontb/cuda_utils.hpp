#pragma once

#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <torch/torch.h>

// Taken from:
// https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/

#define CudaSafeCall(err) fiontb::_CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheck() fiontb::_CudaCheck(__FILE__, __LINE__)

#ifdef __CUDACC__
#define FTB_DEVICE __device__
#define FTB_DEVICE_HOST __device__ __host__
#else
#define FTB_DEVICE
#define FTB_DEVICE_HOST
#endif
namespace fiontb {

inline void _CudaSafeCall(cudaError err, const char *file, const int line) {
  if (err != cudaSuccess) {
    std::stringstream msg;
    msg << "Cuda call error " << file << "(" << line
        << "): " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}
inline void _CudaCheck(const char *file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::stringstream msg;
    msg << "Cuda check error " << file << "(" << line
        << "): " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}

struct CudaKernelDims {
  CudaKernelDims(dim3 grid_dims, dim3 block_dims)
      : grid(grid_dims), block(block_dims) {}

  dim3 grid;
  dim3 block;
};

CudaKernelDims Get1DKernelDims(int size);

CudaKernelDims Get2DKernelDims(int width, int height);

#ifdef __CUDACC__
template <typename scalar_t, unsigned long dims>
using PackedAccessor =
    torch::PackedTensorAccessor<scalar_t, dims, torch::RestrictPtrTraits,
                                size_t>;
#endif

}  // namespace fiontb
