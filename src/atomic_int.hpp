#pragma once

#include <atomic>

#include "device.hpp"

namespace fiontb {
template <Device dev>
struct AtomicInt {
  AtomicInt() : value(nullptr) {}
  inline void operator++() { ++(*value); }
  // std::atomic<int> value;
  void Alloc() {
    value = new int;
    *value = 0;
  }  
  void Free() {
    delete value;
  }
  int32_t get() const { return *value; }
  int32_t *value;
};

template <>
struct AtomicInt<kCUDA> {
  AtomicInt() : value(nullptr) {}

  void Alloc() {
    cudaMalloc((void **)&value, sizeof(int32_t));
    cudaMemset(value, 0, sizeof(int32_t));
  }

  void Free() { cudaFree(value); }
#pragma nv_exec_check_disable
  __device__ inline void operator++() { atomicAdd(value, 1); }

  int32_t get() const {
    int cpu_value;
    cudaMemcpy(&cpu_value, value, sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return cpu_value;
  }
  int32_t *value;
};
}  // namespace fiontb
