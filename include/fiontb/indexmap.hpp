#pragma once

#include <torch/torch.h>

namespace fiontb {
void RasterIndexmap(const torch::Tensor points,
                    const torch::Tensor _proj_matrix, torch::Tensor indexmap,
                    torch::Tensor depth_buffer);
}
