#pragma once

#include <torch/torch.h>
#include "eigen_common.hpp"

namespace fiontb {

#ifdef __CUDACC__
#define FTB_CUDA_HOST_DEVICE __host__ __device__
#else
#define FTB_CUDA_HOST_DEVICE
#endif

inline FTB_CUDA_HOST_DEVICE float GetVectorsAngle(const Eigen::Vector3f &v0,
                                              const Eigen::Vector3f &v1) {
  return acos(v0.dot(v1) / (v0.norm() * v1.norm()));
}

template <typename scalar_t, int size>
using Vector = Eigen::Matrix<scalar_t, size, 1>;


template <typename scalar_t, typename Accessor>
inline FTB_CUDA_HOST_DEVICE Eigen::Matrix<scalar_t, 2, 1> to_vec2(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 2, 1>(acc[0], acc[1], acc[2]);
}

template <typename scalar_t, typename Accessor>
inline FTB_CUDA_HOST_DEVICE Eigen::Matrix<scalar_t, 3, 1> to_vec3(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 3, 1>(acc[0], acc[1], acc[2]);
}

template <typename scalar_t, typename Accessor>
inline FTB_CUDA_HOST_DEVICE Eigen::Matrix<scalar_t, 4, 1> to_vec4(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 4, 1>(acc[0], acc[1], acc[2], acc[3]);
}

}  // namespace fiontb
