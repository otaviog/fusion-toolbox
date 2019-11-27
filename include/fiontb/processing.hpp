#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {

enum EstimateNormalsMethod { kCentralDifferences, kAverage8 };

enum DownsampleXYZMethod { kNearest };

struct Processing {
  static torch::Tensor BilateralDepthFilter(
      const torch::Tensor &input, const torch::Tensor &mask,
      torch::Tensor result, int filter_width = 6, float sigma_d = 4.50000000225,
      float sigma_r = 29.9999880000072, float depth_scale = 1.0f);

  static void EstimateNormals(const torch::Tensor xyz_image,
                              const torch::Tensor mask_image,
                              torch::Tensor out_normals,
                              EstimateNormalsMethod method);

  static void DownsampleXYZ(
      const torch::Tensor &src, const torch::Tensor &mask, float scale,
      torch::Tensor dst, bool normalize = true,
      DownsampleXYZMethod method = DownsampleXYZMethod::kNearest);

  static void DownsampleMask(const torch::Tensor &mask, float scale,
                             torch::Tensor dst);

  static void ErodeMask(const torch::Tensor &in_mask, torch::Tensor out_mask);

  static void RegisterPybind(pybind11::module &m);
};

}  // namespace fiontb
