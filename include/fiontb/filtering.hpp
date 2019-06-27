#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor BilateralFilterDepthImage(torch::Tensor input, torch::Tensor mask,
                                        int filter_width=6,
                                        float sigma_d = 4.50000000225,
                                        float sigma_r = 29.9999880000072,
                                        float depth_scale=1.0f);

}  // namespace fiontb
