#include "filtering.hpp"

#include <numeric>
#include <vector>

namespace fiontb {
torch::Tensor BilateralFilterDepthImage(torch::Tensor input, torch::Tensor mask,
                                        int filter_width, float sigma_color,
                                        float sigma_space) {
  const auto in_acc = input.accessor<int32_t, 2>();
  const auto mask_acc = mask.accessor<uint8_t, 2>();

  const int iwidth = input.size(1);
  const int iheight = input.size(0);

  torch::Tensor result = torch::empty({iheight, iwidth}, torch::kInt32);
  auto re_acc = result.accessor<int32_t, 2>();

  const int half_width = filter_width / 2;

  const float inv_sigma_color_sqr = 1.0 / (sigma_color * sigma_color);
  const float inv_sigma_space_sqr = 1.0 / (sigma_space * sigma_space);

  for (int row = 0; row < input.size(0); ++row) {
    for (int col = 0; col < input.size(1); ++col) {
      re_acc[row][col] = 0;
      if (mask_acc[row][col] == 0) continue;

      const float depth = in_acc[row][col]*0.001f;

      float color_sum = 0.0f;
      float weight_sum = 0.0f;

      for (int y = -half_width; y <= half_width; ++y) {
        const int crow = row + y;
        if (crow < 0 || crow >= iheight) continue;

        for (int x = -half_width; x <= half_width; ++x) {
          const int ccol = col + x;
          if (ccol < 0 || ccol >= iwidth) continue;

          if (mask_acc[crow][ccol] == 0) continue;

          const float curr_depth = in_acc[crow][ccol]*0.001;

          const float dx = col - ccol;
          const float dy = row - crow;
          const float space_sqr = dx * dx + dy * dy;
          
          const float dcolor = depth - curr_depth;
          const float color_sqr = dcolor*dcolor;

          const float weight =
              std::exp(-0.5f * (space_sqr * inv_sigma_space_sqr +
                                color_sqr * inv_sigma_color_sqr));
          color_sum += curr_depth * weight;
          weight_sum += weight;
        }
      }

      if (weight_sum > 0.0f) {
        re_acc[row][col] = int32_t((color_sum / weight_sum)*1000.0f);
      }
    }
  }
  return result;
}

}  // namespace fiontb
