#pragma once

#include <math_constants.h>
#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"

namespace fiontb {

inline FTB_DEVICE_HOST float GetVectorsAngle(const Eigen::Vector3f &v0,
                                             const Eigen::Vector3f &v1) {
  return acos(v0.dot(v1) / (v0.norm() * v1.norm()));
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

}  // namespace fiontb
