#pragma once

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace slamtb {

template <Device dev>
struct MergeMap {};

template <>
struct MergeMap<kCPU> {
  torch::TensorAccessor<int64_t, 2> merge_map;
  static std::mutex mutex_;  // Slowest, but easy to test.

  static inline int32_t float_as_int(float in) {
    union {
      int32_t i;
      float f;
    } s;
    s.f = in;
    return s.i;
  }

  static inline float int_as_float(int32_t in) {
    union {
      int32_t i;
      float f;
    } s;
    s.i = in;
    return s.f;
  }

  MergeMap(torch::Tensor &merge_map)
      : merge_map(merge_map.accessor<int64_t, 2>()) {}

  void Set(int row, int col, float dist, int32_t index) {
    std::lock_guard<std::mutex> lock(mutex_);

    typedef unsigned long long int UInt64;
    const UInt64 this_value = UInt64(index) << 32 | UInt64(float_as_int(dist));
    
    UInt64 *address =
        (UInt64 *)(merge_map.data() + row * merge_map.stride(0) + col);
    UInt64 curr = *address;

    if (dist < int_as_float(curr & 0xffffffff)) {
      *address = this_value;
    }
  }
};

#ifdef __CUDACC__
template <>
struct MergeMap<kCUDA> {
  torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> merge_map;

  MergeMap(torch::Tensor merge_map)
      : merge_map(
            merge_map
                .packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>()) {}

  __device__ void Set(int row, int col, float dist, int32_t index) {
    typedef unsigned long long int UInt64;

    const UInt64 this_value = UInt64(index) << 32 | UInt64(__float_as_int(dist));

    UInt64 *address =
        (UInt64 *)(merge_map.data() + row * merge_map.stride(0) + col);

    volatile UInt64 old = *address;
    UInt64 assumed;
    do {
      assumed = old;

      if (dist >= __int_as_float(assumed & 0xffffffff)) break;

      old = atomicCAS(address, assumed, this_value);
    } while (assumed != old);
  }
};
#endif

template <Device dev>
struct MergeMapAccessor {
  const typename Accessor<dev, int64_t, 2>::T merge_map;

  MergeMapAccessor(const torch::Tensor &merge_map)
      : merge_map(Accessor<dev, int64_t, 2>::Get(merge_map)) {}

  FTB_DEVICE_HOST int32_t operator()(int row, int col) const {
    return merge_map[row][col] >> 32;
  }
};

}  // namespace slamtb
