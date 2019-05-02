#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor FilterSearch(torch::Tensor dist_mtx, torch::Tensor idx_mtx,
                           torch::Tensor live_normals,
                           torch::Tensor model_normals, float min_dist,
                           float min_normal_dot);
}
