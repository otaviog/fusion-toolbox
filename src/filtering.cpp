#include "filtering.hpp"

#include <numeric>

namespace fiontb {

template <typename scalar_t>
torch::Tensor BilateralFilterDepthImage_cpu_kernel(
    const torch::Tensor input, const torch::Tensor mask, int filter_width,
    float sigma_color, float sigma_space, float depth_scale) {
  const auto in_acc = input.accessor<scalar_t, 2>();
  const auto mask_acc = mask.accessor<uint8_t, 2>();

  const int iwidth = input.size(1);
  const int iheight = input.size(0);

  torch::Tensor result = torch::empty({iheight, iwidth}, input.dtype());
  auto re_acc = result.accessor<scalar_t, 2>();

  const int half_width = filter_width / 2;

  const float inv_sigma_color_sqr = 1.0 / (sigma_color * sigma_color);
  const float inv_sigma_space_sqr = 1.0 / (sigma_space * sigma_space);
  const float inv_depth_scale = 1.0 / depth_scale;

#pragma omp parallel for
  for (int row = 0; row < iheight; ++row) {
    for (int col = 0; col < iwidth; ++col) {
      re_acc[row][col] = 0;
      if (mask_acc[row][col] == 0) continue;

      const float depth = float(in_acc[row][col]) * depth_scale;

      float color_sum = 0.0f;
      float weight_sum = 0.0f;

      for (int y = -half_width; y <= half_width; ++y) {
        const int crow = row + y;
        if (crow < 0 || crow >= iheight) continue;

        for (int x = -half_width; x <= half_width; ++x) {
          const int ccol = col + x;
          if (ccol < 0 || ccol >= iwidth) continue;

          if (mask_acc[crow][ccol] == 0) continue;

          const float curr_depth = in_acc[crow][ccol] * depth_scale;

          const float dx = col - ccol;
          const float dy = row - crow;
          const float space_sqr = dx * dx + dy * dy;

          const float dcolor = depth - curr_depth;
          const float color_sqr = dcolor * dcolor;

          const float weight =
              std::exp(-0.5f * (space_sqr * inv_sigma_space_sqr +
                                color_sqr * inv_sigma_color_sqr));
          color_sum += curr_depth * weight;
          weight_sum += weight;
        }
      }

      if (weight_sum > 0.0f) {
        re_acc[row][col] = scalar_t((color_sum / weight_sum) * inv_depth_scale);
      }
    }
  }
  return result;
}

torch::Tensor BilateralFilterDepthImage_gpu(const torch::Tensor input,
                                            const torch::Tensor mask,
                                            int filter_width, float sigma_color,
                                            float sigma_space,
                                            float depth_scale);

torch::Tensor BilateralFilterDepthImage(torch::Tensor input, torch::Tensor mask,
                                        int filter_width, float sigma_color,
                                        float sigma_space, float depth_scale) {
  if (input.is_cuda()) {
    return BilateralFilterDepthImage_gpu(input, mask, filter_width, sigma_color,
                                         sigma_space, depth_scale);
  } else {
    return AT_DISPATCH_ALL_TYPES(
        input.scalar_type(), "BilateralFilterDepthImage_cpu_kernel", ([&] {
          return BilateralFilterDepthImage_cpu_kernel<scalar_t>(
              input, mask, filter_width, sigma_color, sigma_space, depth_scale);
        }));
  }
}

}  // namespace fiontb
