#pragma once

#include <torch/torch.h>

namespace fiontb {

/**
 * Differentiable SO3 + translation exp function.
 */
struct SO3tExpOp {
  static torch::Tensor Forward(torch::Tensor tangent);
  static torch::Tensor Backward(const torch::Tensor &dy_matrices,
                                const torch::Tensor &x_upsilon_omegas,
                                const torch::Tensor &y_matrices);
};
}  // namespace fiontb
