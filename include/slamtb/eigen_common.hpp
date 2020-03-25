#pragma once

#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Eigen>
#pragma GCC diagnostic pop

#include <torch/torch.h>
#include "cuda_utils.hpp"

namespace slamtb {
inline Eigen::Vector3f from_tensorv3f(const torch::Tensor &tensor) {
  auto acs = tensor.accessor<float, 1>();
  return Eigen::Vector3f(acs[0], acs[1], acs[2]);
}

template <typename scalar_t, int size>
using Vector = Eigen::Matrix<scalar_t, size, 1>;

template <typename scalar_t, typename Accessor>
inline FTB_DEVICE_HOST Eigen::Matrix<scalar_t, 2, 1> to_vec2(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 2, 1>(acc[0], acc[1], acc[2]);
}

template <typename scalar_t, typename Accessor>
inline FTB_DEVICE_HOST Eigen::Matrix<scalar_t, 3, 1> to_vec3(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 3, 1>(acc[0], acc[1], acc[2]);
}

template <typename scalar_t, typename Accessor>
inline FTB_DEVICE_HOST Eigen::Matrix<scalar_t, 4, 1> to_vec4(
    const Accessor acc) {
  return Eigen::Matrix<scalar_t, 4, 1>(acc[0], acc[1], acc[2], acc[3]);
}

template <typename scalar_t, int rows, int cols, typename Accessor>
inline FTB_DEVICE_HOST Eigen::Matrix<scalar_t, rows, cols> to_matrix(
    const Accessor acc) {
  Eigen::Matrix<scalar_t, rows, cols> mat;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat(i, j) = acc[i][j];
    }
  }

  return mat;
}
}  // namespace slamtb
