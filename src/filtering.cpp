#include "filtering.hpp"

#include <numeric>
#include <vector>

namespace fiontb {
torch::Tensor FilterDepthImage(torch::Tensor input, torch::Tensor mask,
                               torch::Tensor kernel) {
  const auto in_acc = input.accessor<int16_t, 2>();
  const auto kr_acc = kernel.accessor<int16_t, 2>();
  const auto mask_acc = mask.accessor<uint8_t, 2>();

  const int iwidth = input.size(1);
  const int iheight = input.size(0);

  const int kwidth = kernel.size(1);
  const int kheight = kernel.size(0);

  std::vector<long> values;
  values.reserve(kwidth * kheight);

  torch::Tensor result = torch::empty({iheight, iwidth}, torch::kInt16);
  auto re_acc = result.accessor<int16_t, 2>();

  for (int row = 0; row < input.size(0); ++row) {
    for (int col = 0; col < input.size(1); ++col) {
      re_acc[row][col] = 0;
      if (mask_acc[row][col] == 0) continue;

      values.clear();

      for (int krow = 0; krow < kernel.size(0); ++krow) {
        for (int kcol = 0; kcol < kernel.size(1); ++kcol) {
          const int y = row - (krow - kheight / 2);
          if (y < 0 || y >= iheight) continue;

          const int x = col - (kcol - kwidth / 2);
          if (x < 0 || x >= iwidth) continue;

          if (mask_acc[y][x] == 0) continue;

          if (in_acc[y][x] == 0) continue;
          values.push_back(in_acc[y][x]);
        }
      }

      const long accum = std::accumulate(values.begin(), values.end(), 0);

      const float mean = static_cast<float>(double(accum) / values.size());

      float std_accum = 0;
      for (long value : values) {
        std_accum = (float(value) - mean)*(float(value) - mean);
      }

      const long stdv = std::sqrt(std_accum / values.size()) + 1;
      
      long cvalue = in_acc[row][col];
      
      long filter_accum = 0;
      int neighbor_count = 0;
      for (int krow = 0; krow < kernel.size(0); ++krow) {
        for (int kcol = 0; kcol < kernel.size(1); ++kcol) {
          const int y = row - (krow - kheight / 2);
          if (y < 0 || y >= iheight) continue;

          const int x = col - (kcol - kwidth / 2);
          if (x < 0 || x >= iwidth) continue;

          if (mask_acc[y][x] == 0) continue;

          const long im_value = long(in_acc[y][x]);

          if (im_value == 0) continue;
          if (im_value < cvalue - stdv || im_value > cvalue + stdv) continue;
          
          const float kr_value = kr_acc[krow][kcol];

          filter_accum += im_value * kr_value;
          ++neighbor_count;
        }
      }

      re_acc[row][col] = int16_t(double(filter_accum) / neighbor_count);
    }
  }
  return result;
}
}  // namespace fiontb
