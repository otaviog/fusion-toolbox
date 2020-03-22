#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {

/**
 * Differentiable exponential rotation and translation layer
 * operations.  See the compact formula of Gallego, Guillermo, and
 * Anthony Yezzi. "A compact formula for the derivative of a 3-D
 * rotation in exponential coordinates." Journal of Mathematical
 * Imaging and Vision 51, no. 3 (2015): 378-384.
 *
 */
struct ExpRtToMatrixOp {
  /**
   * Forward, called from Python.
   *
   * @param[in] exp_rt [Bx6] tensor containing the translation vector
   *  and 3 exponetial coordinates.
   *
   * @param[out] matrix [Bx12] rigid transformation matrix.
   */
  static void Forward(const torch::Tensor &exp_rt, torch::Tensor matrix);

  /**
   * Backward, called from Python.
   *
   * @param[in] d_matrix_loss [Bx3x4] The matrix gradient w.r.t. to final
   * loss. That is \f$ = \frac{\delta a_ij}{\delta L} \f$.

   * @param[in] exp_rt [Bx6] The original translation vector and
   *  exponetial coordinates.
   *
   * @param[in] y_matrix [Bx3x4] The resulting matrix from the
   * forward pass.
   *
   * @param[out] d_exp_rt_loss [Bx6] The translation and exponential
   * rotation gradient w.r.t. the loss function. That is \f$ =
   * \frac{\delta w_i}{\delta \delta L}\f$
   */
  static void Backward(const torch::Tensor &d_matrix_loss,
                       const torch::Tensor &exp_rt,
                       const torch::Tensor &y_matrix,
                       torch::Tensor d_exp_rt_loss);

  /**
   * Register the operations.
   */
  static void RegisterPybind(pybind11::module &m);
};

/**
 * Converts a rigid transformation matrix into a translation and
 * exponential rotation.
 */
struct MatrixToExpRtOp {
  /**
   * Forward, called from Python.
   *
   * @param[in] matrix [Bx3x4] tensors.
   *
   * @param[out] exp_rt [Bx6] tensors. XYZ translation and 3
   * exponential rotation elements.
   */
  static void Forward(const torch::Tensor &matrix, torch::Tensor exp_rt);

  /**
   * Register the operations.
   */
  static void RegisterPybind(pybind11::module &m);
};

}  // namespace fiontb
