#include "filtering.hpp"

#include "cuda_error.hpp"

namespace fiontb {

template <typename scalar_t>
__global__ void BilateralFilterDepthImage_gpu_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits,
                                      size_t>
        input,
    const torch::PackedTensorAccessor<uint8_t, 2, torch::RestrictPtrTraits,
                                      size_t>
        mask,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
        result,
    int half_width, float inv_sigma_color_sqr, float inv_sigma_space_sqr) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = input.size(1);
  const int height = input.size(0);
  if (row >= height || col >= width) {
    return;
  }

  result[row][col] = 0;
  if (mask[row][col] == 0) return;

  const scalar_t depth = input[row][col];

  float color_sum = 0.0f;
  float weight_sum = 0.0f;

  for (int y = -half_width; y <= half_width; ++y) {
    const int crow = row + y;
    if (crow < 0 || crow >= height) continue;

    for (int x = -half_width; x <= half_width; ++x) {
      const int ccol = col + x;
      if (ccol < 0 || ccol >= width) continue;

      if (mask[crow][ccol] == 0) continue;

      const scalar_t curr_depth = input[crow][ccol];

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
    result[row][col] = scalar_t(color_sum / weight_sum);
  }
}

torch::Tensor BilateralFilterDepthImage_gpu(const torch::Tensor input,
                                            const torch::Tensor mask,
                                            int filter_width,
                                            float sigma_color,
                                            float sigma_space) {
  const int width = input.size(1);
  const int height = input.size(0);

  dim3 block_dim = dim3(16, 16);

  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);
  torch::Tensor result =
      torch::empty({height, width}, torch::TensorOptions()
                   .dtype(input.scalar_type())
                   .device(torch::kCUDA, 0));
  const int half_width = filter_width / 2;
  const float inv_sigma_color_sqr = 1.0 / (sigma_color * sigma_color);
  const float inv_sigma_space_sqr = 1.0 / (sigma_space * sigma_space);

  AT_DISPATCH_ALL_TYPES(
      input.scalar_type(), "BilateralFilterDepthImage_gpu", ([&] {
        BilateralFilterDepthImage_gpu_kernel<<<grid_size, block_dim>>>(
            input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits,
                                  size_t>(),
            mask.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits,
                                 size_t>(),
            result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits,
            size_t>(),
            half_width, inv_sigma_color_sqr, inv_sigma_space_sqr);
      }));
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
  return result;
}

}  // namespace fiontb