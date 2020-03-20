#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {

/**
 * Differentiable exp rotation and translation layer
 */
struct ExpRtToMatrixOp {
  static torch::Tensor Forward(torch::Tensor exp_rt);
  static torch::Tensor Backward(const torch::Tensor &dy_matrices,
                                const torch::Tensor &x_upsilon_omegas,
                                const torch::Tensor &y_matrices);
  static void RegisterPybind(pybind11::module &m);
};

/**
 * Differentiable exp rotation and translation layer
 */
struct ExpRtTransformOp {
  static void Forward(const torch::Tensor &exp_rt,
                      const torch::Tensor &x_points, torch::Tensor y_points);
  static void Backward(const torch::Tensor &x_exp_rt,
                       const torch::Tensor &x_points,
                       const torch::Tensor &dy_points,
                       torch::Tensor dx_exp_rt);
  static void RegisterPybind(pybind11::module &m);
};

/**
 * Differentiable quaternion + 3D layer
 */
struct QuatRtTransformOp {
  static void Forward(const torch::Tensor &x_quat_t,
                      const torch::Tensor &x_points, torch::Tensor y_points);
  static void Backward(const torch::Tensor &x_quat_t,
                       const torch::Tensor &x_points,
                       const torch::Tensor &dy_points, torch::Tensor dx_quat_t);
  static void RegisterPybind(pybind11::module &m);
};

}  // namespace fiontb
