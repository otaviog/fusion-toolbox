#pragma once

#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Eigen>
#pragma GCC diagnostic pop

#include <torch/torch.h>

namespace fiontb {
inline Eigen::Vector3f from_tensorv3f(const torch::Tensor &tensor) {
  auto acs = tensor.accessor<float, 1>();
  return Eigen::Vector3f(acs[0], acs[1], acs[2]);
}
}  // namespace fiontb
