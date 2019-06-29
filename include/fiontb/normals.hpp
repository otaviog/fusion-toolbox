#pragma once

#include <torch/torch.h>

namespace fiontb {
enum EstimateNormalsMethod { kCentralDifferences, kAverage8 };

void EstimateNormals(const torch::Tensor xyz_image,
                     const torch::Tensor mask_image, torch::Tensor out_normals,
                     EstimateNormalsMethod method);
}  // namespace fiontb
