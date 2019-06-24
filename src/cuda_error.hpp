#pragma once

#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

// Taken from:
// https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/

#define CudaSafeCall(err) fiontb::_CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheck() fiontb::_CudaCheck(__FILE__, __LINE__)

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
}  // namespace fiontb
