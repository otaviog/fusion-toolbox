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



}  // namespace fiontb
