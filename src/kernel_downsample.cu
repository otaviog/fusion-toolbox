#include "downsample.hpp"

#include "cuda_utils.hpp"
#include "math.hpp"

namespace fiontb {
__global__ void DownsampleXYZNearest_gpu_kernel(
    const PackedAccessor<float, 3> xyz_img,
    const PackedAccessor<uint8_t, 2> mask, float inv_scale,
    PackedAccessor<float, 3> dst) {
  const int dst_width = dst.size(1);
  const int dst_height = dst.size(0);

  const int dst_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int dst_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (dst_row >= dst_height || dst_col >= dst_width) return;

  const int center_src_row = dst_row * inv_scale,
            center_src_col = dst_col * inv_scale;

  const int src_width = xyz_img.size(1), src_height = xyz_img.size(0);
  const int where[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  Eigen::Vector3f xyz_mean;
  int xyz_count = 0;

  for (int i = 0; i < 4; ++i) {
    const int src_row = center_src_row + where[i][0];
    const int src_col = center_src_col + where[i][1];

    if (src_col < 0 || src_col >= src_width) continue;
    if (mask[src_row][src_col] == 0) continue;

    const Eigen::Vector3f src_xyz = to_vec3<float>(xyz_img[src_row][src_col]);
    xyz_mean += src_xyz;
    ++xyz_count;
  }

  xyz_mean /= float(xyz_count);

  Eigen::Vector3f nearest_xyz(0, 0, 0);
  float best_dist = 99999.0f;

  for (int i = 0; i < 4; ++i) {
    const int src_row = center_src_row + where[i][0];
    const int src_col = center_src_col + where[i][1];

    if (src_col < 0 || src_col >= src_width) continue;
    if (mask[src_row][src_col] == 0) continue;

    const Eigen::Vector3f src_xyz = to_vec3<float>(xyz_img[src_row][src_col]);
    const float dist = (src_xyz - xyz_mean).squaredNorm();
    if (dist < best_dist) {
      best_dist = dist;
      nearest_xyz = src_xyz;
    }
  }

  dst[dst_row][dst_col][0] = nearest_xyz[0];
  dst[dst_row][dst_col][1] = nearest_xyz[1];
  dst[dst_row][dst_col][2] = nearest_xyz[2];
}

void DownsampleXYZ(const torch::Tensor xyz_image, const torch::Tensor mask,
                   float scale, torch::Tensor dst, bool normalize,
                   DownsampleXYZMethod method) {
  const CudaKernelDims kern_dims =
      Get2DKernelDims(xyz_image.size(1), xyz_image.size(0));

  DownsampleXYZNearest_gpu_kernel<<<kern_dims.grid, kern_dims.block>>>(
      xyz_image.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      mask.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits, size_t>(),
      1.0f / scale,
      dst.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
}

}  // namespace fiontb