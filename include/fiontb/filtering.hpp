#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor BilateralDepthFilter(const torch::Tensor &input,
                                   const torch::Tensor &mask,
                                   torch::Tensor result, int filter_width = 6,
                                   float sigma_d = 4.50000000225,
                                   float sigma_r = 29.9999880000072,
                                   float depth_scale = 1.0f);

struct FeatureMapOp {
  static void Forward(const torch::Tensor feature_map, const torch::Tensor uv,
                      torch::Tensor out_features, torch::Tensor out_bound_mask);

  static void Backward(const torch::Tensor feature_map, const torch::Tensor uv,
                       const torch::Tensor dl_value, torch::Tensor dl_uv);
};

void CalcSobelGradient(const torch::Tensor image, torch::Tensor out_grad);

}  // namespace fiontb
