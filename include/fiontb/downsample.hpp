#pragma once

#include <torch/torch.h>

namespace fiontb {

enum DownsampleXYZMethod { kNearest };

void DownsampleXYZ(const torch::Tensor src, const torch::Tensor mask,
                   float scale, torch::Tensor dst, bool normalize = true,
                   DownsampleXYZMethod method = DownsampleXYZMethod::kNearest);
}  // namespace fiontb
