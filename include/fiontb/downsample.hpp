#pragma once

#include <torch/torch.h>

namespace fiontb {

enum DownsampleXYZMethod { kNearest };

struct Downsample {
  static void DownsampleXYZ(
      const torch::Tensor &src, const torch::Tensor &mask, float scale,
      torch::Tensor dst, bool normalize = true,
      DownsampleXYZMethod method = DownsampleXYZMethod::kNearest);

  static void DownsampleMask(const torch::Tensor &mask, float scale,
                             torch::Tensor dst);
};

}  // namespace fiontb
