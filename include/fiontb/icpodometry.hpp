#pragma once

#include <vector>

#include <torch/torch.h>

namespace fiontb {

class ICPOdometry {
 public:
  ICPOdometry(std::vector<float> scales, std::vector<int> num_iters);

  torch::Tensor Estimate(torch::Tensor points0, torch::Tensor normals0,
                         torch::Tensor points1, torch::Tensor kcam,
                         torch::Tensor init_transf);

 private:
  std::vector<float> scales_;
  std::vector<int> num_iters_;
};

void EstimateJacobian_gpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor points1, const torch::Tensor kcam,
                          const torch::Tensor params, torch::Tensor jacobian,
                          torch::Tensor residual);

}  // namespace fiontb
