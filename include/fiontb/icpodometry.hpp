#pragma once

#include <vector>

#include <torch/torch.h>

namespace fiontb {

void EstimateJacobian_gpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor mask0,
                          const torch::Tensor points1,
                          const torch::Tensor mask1,
                          const torch::Tensor kcam,
                          const torch::Tensor params, torch::Tensor jacobian,
                          torch::Tensor residual);

}  // namespace fiontb
