#include "filtering.hpp"

#include "accessor.hpp"
#include "cuda_utils.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {

template <Device dev, typename scalar_t>
struct BilateralDepthFilterKernel {
  const typename Accessor<dev, scalar_t, 2>::T input;
  const typename Accessor<dev, bool, 2>::T mask;
  typename Accessor<dev, scalar_t, 2>::T result;
  int half_width;
  float inv_sigma_color_sqr, inv_sigma_space_sqr, depth_scale;

  BilateralDepthFilterKernel(const torch::Tensor input,
                             const torch::Tensor mask, torch::Tensor result,
                             int half_width, float inv_sigma_color_sqr,
                             float inv_sigma_space_sqr, float depth_scale)
      : input(Accessor<dev, scalar_t, 2>::Get(input)),
        mask(Accessor<dev, bool, 2>::Get(mask)),
        result(Accessor<dev, scalar_t, 2>::Get(result)),
        half_width(half_width),
        inv_sigma_color_sqr(inv_sigma_color_sqr),
        inv_sigma_space_sqr(inv_sigma_space_sqr),
        depth_scale(depth_scale) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    result[row][col] = 0;
    if (mask[row][col] == 0) return;

    const float depth = input[row][col] * depth_scale;

    float color_sum = 0.0f;
    float weight_sum = 0.0f;
    const float inv_depth_scale = 1.0 / depth_scale;

    const int height = input.size(0);
    const int width = input.size(1);

    for (int y = -half_width; y <= half_width; ++y) {
      const int crow = row + y;
      if (crow < 0 || crow >= height) continue;

      for (int x = -half_width; x <= half_width; ++x) {
        const int ccol = col + x;
        if (ccol < 0 || ccol >= width) continue;

        if (mask[crow][ccol] == 0) continue;

        const float curr_depth = input[crow][ccol] * depth_scale;

        const float dx = col - ccol;
        const float dy = row - crow;
        const float space_sqr = dx * dx + dy * dy;

        const float dcolor = depth - curr_depth;
        const float color_sqr = dcolor * dcolor;

        const float weight = expf(-0.5f * (space_sqr * inv_sigma_space_sqr +
                                           color_sqr * inv_sigma_color_sqr));
        color_sum += curr_depth * weight;
        weight_sum += weight;
      }
    }

    if (weight_sum > 0.0f) {
      result[row][col] = scalar_t((color_sum / weight_sum) * inv_depth_scale);
    }
  }
};

torch::Tensor BilateralDepthFilter(const torch::Tensor &input,
                                   const torch::Tensor &mask,
                                   torch::Tensor result, int filter_width,
                                   float sigma_color, float sigma_space,
                                   float depth_scale) {
  const int width = input.size(1);
  const int height = input.size(0);

  const int half_width = filter_width / 2;
  const float inv_sigma_color_sqr = 1.0 / (sigma_color * sigma_color);
  const float inv_sigma_space_sqr = 1.0 / (sigma_space * sigma_space);

  if (input.is_cuda()) {
    FTB_CHECK(mask.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(result.is_cuda(), "Expected a cuda tensor");

    AT_DISPATCH_ALL_TYPES(
        input.scalar_type(), "BilateralFilterDepthImage_gpu", ([&] {
          BilateralDepthFilterKernel<kCUDA, scalar_t> kernel(
              input, mask, result, half_width, inv_sigma_color_sqr,
              inv_sigma_space_sqr, depth_scale);
          Launch2DKernelCUDA(kernel, input.size(1), input.size(0));
        }));
  } else {
    FTB_CHECK(!mask.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!result.is_cuda(), "Expected a cpu tensor");

    AT_DISPATCH_ALL_TYPES(
        input.scalar_type(), "BilateralFilterDepthImage_cpu", ([&] {
          BilateralDepthFilterKernel<kCPU, scalar_t> kernel(
              input, mask, result, half_width, inv_sigma_color_sqr,
              inv_sigma_space_sqr, depth_scale);
          Launch2DKernelCPU(kernel, input.size(1), input.size(0));
        }));
  }
  return result;
}

}  // namespace fiontb