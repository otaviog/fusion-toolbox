#pragma once

#include <torch/torch.h>

namespace fiontb {

struct Matching {
  static void MatchDensePoints(const torch::Tensor target_points,
                        const torch::Tensor target_mask,
                        const torch::Tensor source_points,
                        const torch::Tensor kcam, const torch::Tensor rt_cam,
                        torch::Tensor out_point, torch::Tensor out_index);
};
}
