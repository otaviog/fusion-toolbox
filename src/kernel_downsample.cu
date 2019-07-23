#include "filtering.hpp"

#include "cuda_utils.hpp"

namespace fiontb {
__global__ void DownsampleXYZNearest_gpu_kernel(
    const PackedAccessor<float, 3> xyz_img,
    const PackedAccessor<uint8_t, 2> mask, float inv_scale,
    PackedAccessor<float, 3> result) {
  const int dst_width = result.size(1);
  const int dst_height = result.size(0);

  const int dst_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int dst_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (dst_row >= dst_height || dst_col >= dst_width) return;

  float best_dist = 99999.0f;

  const int ct_src_row = dst_row * inv_scale;
  const int ct_src_col = dst_col * inv_scale;

  const int src_width = xyz_img.size(1);
  const int src_height = xyz_img.size(0);

  Eigen::Vector3f xyz_mean;
  int xyz_count = 0;

  for (int y = -half_width; y <= half_width; y++) {
    const int src_row = ct_src_row + y;
    if (src_row < 0 || src_row >= src_height) continue;
    for (int x = -half_width; x <= half_width; x++) {
      const int src_col = ct_src_col + x;

      if (src_col < 0 || src_col >= src_width) continue;
      if (mask[src_row][src_col] == 0) continue;

      const Eigen::Vector3f src_xyz = to_vec3<float>(xyz_img[src_row][src_col]);
      xyz_mean += src_xyz;
      ++xyz_count;
    }
  }

  float nearest_dist = CUDART_INF_F;
  Eigen::Vector3f nearest_xyz(0.0f, 0.0f, 0.0f);
  for (int y = -half_width; y <= half_width; y++) {
    const int src_row = ct_src_row + y;
    if (src_row < 0 || src_row >= src_height) continue;
    for (int x = -half_width; x <= half_width; x++) {
      const int src_col = ct_src_col + x;

      if (src_col < 0 || src_col >= src_width) continue;
      if (mask[src_row][src_col] == 0) continue;

      const Eigen::Vector3f src_xyz = to_vec3<float>(xyz_img[src_row][src_col]);
      const float dist = (src_xyz - xyz_mean).squaredNorm();
      if (dist < best_dist) {
        best_dist = dist;
        nearest_xyz = src_syz;
      }
    }
  }

  result[dst_row][dst_col] = nearest_xyz;
}

void DownsampleXYZ(const torch::Tensor xyz_image, const torch::Tensor mask,
                   float scale, torch::Tensor result,
                   DownsampleXYZMethod method) {
  const CudaKernelDims kern_dims =
      Get2DKernelDims(xyz_image.size(1), xyz_image.size(0));

  DownsampleXYZNearest_gpu_kernel(
      xyz_image.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      mask.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits, size_t>(),
      1.0f / scale result
                 .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>())
}

}  // namespace fiontb