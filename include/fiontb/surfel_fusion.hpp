#pragma once

#include <torch/torch.h>

namespace fiontb {
void CarveSpace(const torch::Tensor stable_pos_fb,
                const torch::Tensor stable_idx_fb,
                const torch::Tensor view_pos_fb,
                const torch::Tensor view_idx_fb, torch::Tensor mask,
                int neighbor_size);
}
