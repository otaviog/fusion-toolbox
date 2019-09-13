#pragma once

#include <math_constants.h>
#include <limits>

#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "device.hpp"
#include "eigen_common.hpp"

namespace fiontb {

inline FTB_DEVICE_HOST float GetVectorsAngle(const Eigen::Vector3f &v0,
                                             const Eigen::Vector3f &v1) {
  return acos(v0.dot(v1) / (v0.norm() * v1.norm()));
}

template <typename scalar_t>
inline FTB_DEVICE_HOST Vector<scalar_t, 3> GetNormal(
    const Vector<scalar_t, 3> &p0, const Vector<scalar_t, 3> &p1,
    const Vector<scalar_t, 3> &p2) {
  return (p1 - p0).cross(p2 - p0).normalized();
}

template <typename scalar_t>
inline FTB_DEVICE_HOST Eigen::Matrix<scalar_t, 3, 3> SkewMatrix(
    const Vector<scalar_t, 3> &v) {
  Eigen::Matrix<scalar_t, 3, 3> skew;
  // clang-format off
  skew << 
      0, -v[2], v[1],
      v[2], 0, -v[0],
      -v[1], v[0], 0;
  // clang-format on

  return skew;
}

template <Device dev, typename scalar_t>
struct NumericLimits {};

#ifdef __CUDACC__
template <>
struct NumericLimits<kCUDA, float> {
  static inline __device__ float infinity() noexcept { return CUDART_INF_F; }
};

template <>
struct NumericLimits<kCUDA, double> {
  static inline __device__ float infinity() noexcept { return CUDART_INF; }
};
#endif

template <typename scalar_t>
struct NumericLimits<kCPU, scalar_t> {
  static constexpr scalar_t infinity() noexcept {
    return std::numeric_limits<scalar_t>::infinity();
  }
};

}  // namespace fiontb
