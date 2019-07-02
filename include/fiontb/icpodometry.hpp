#pragma once

#include <vector>

#include <torch/torch.h>

namespace fiontb {

class ICPOdometry {
 public:
  ICPOdometry(std::vector<float> scales, std::vector<int> num_iters);
  
  torch::Tensor Estimate(torch::Tensor points0, torch::Tensor normals0,
                         torch::Tensor points1, torch::Tensor init_transf);
 private:
  std::vector<float> scales_;
  std::vector<int> num_iters_;
};

}
