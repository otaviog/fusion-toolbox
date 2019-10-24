#pragma once

#include <atomic>

#include "device.hpp"

namespace fiontb {

template <Device dev>
struct AtomicInt {
  AtomicInt() : value_(nullptr) {}
  inline void operator++() { ++(*value_); }
  // std::atomic<int> value_;
  void Alloc() {
    value_ = new int;
    *value_ = 0;
  }
  void Free() { delete value_; }
  int32_t value() const { return *value_; }
  int32_t *value_;
};

template <>
struct AtomicInt<kCUDA> {
  AtomicInt() : value_(nullptr) {}

  void Alloc() {
    cudaMalloc((void **)&value_, sizeof(int32_t));
    cudaMemset(value_, 0, sizeof(int32_t));
  }

  void Free() { cudaFree(value_); }
#pragma nv_exec_check_disable
  __device__ inline void operator++() { atomicAdd(value_, 1); }

  int32_t value() const {
    int cpu_value;
    cudaMemcpy(&cpu_value, value_, sizeof(int32_t),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return cpu_value;
  }

  int32_t *value_;
};

template <Device dev>
class ScopedAtomicInt {
 public:
  ScopedAtomicInt() { instance.Alloc(); }

  ~ScopedAtomicInt() { instance.Free(); }

  ScopedAtomicInt(const ScopedAtomicInt &) = delete;

  ScopedAtomicInt &operator=(const ScopedAtomicInt &) = delete;

  AtomicInt<dev> get() const { return instance; }

  operator int() const { return instance.value(); }

 private:
  AtomicInt<dev> instance;
};

}  // namespace fiontb
