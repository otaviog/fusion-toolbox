#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor CalculateFrameNormals(const torch::Tensor xyz_image,
                                    const torch::Tensor mask_image);
}
