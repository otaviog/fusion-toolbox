#pragma once

#include <torch/torch.h>

namespace fiontb {
torch::Tensor FilterDepthImage(torch::Tensor input, torch::Tensor mask,
                               torch::Tensor kernel);
}
