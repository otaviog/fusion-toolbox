#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "math.hpp"

namespace fiontb {
namespace {
__global__ void FindMergeable_gpu_kernel(
    const PackedAccessor<float, 3> pos_fb,
    const PackedAccessor<float, 3> normal_rad_fb,
    const PackedAccessor<int32_t, 3> idx_fb,
    PackedAccessor<int64_t, 2> merge_map, float max_dist, float max_angle,
    int neighbor_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = pos_fb.size(1);
  const int height = pos_fb.size(0);

  if (row >= height || col >= width) return;

  merge_map[row][col] = -1;
  if (idx_fb[row][col][1] == 0) return;

  const Eigen::Vector3f pos(pos_fb[row][col][0], pos_fb[row][col][1],
                            pos_fb[row][col][2]);
  const Eigen::Vector3f normal(normal_rad_fb[row][col][0],
                               normal_rad_fb[row][col][1],
                               normal_rad_fb[row][col][2]);
  const float radius = normal_rad_fb[row][col][3];

  int nearest_fb_idx = -1;
  float nearest_dist = max_dist * max_dist;
  int count = 0;

  for (int krow = max(0, row - neighbor_size);
       krow < min(height - 1, row + neighbor_size); ++krow) {
    for (int kcol = max(0, col - neighbor_size);
         kcol < min(width - 1, col + neighbor_size); ++kcol) {
      if (krow == row && kcol == col) continue;
      if (idx_fb[krow][kcol][1] == 0) continue;

      const Eigen::Vector3f neighbor_pos(
          pos_fb[krow][kcol][0], pos_fb[krow][kcol][1], pos_fb[krow][kcol][2]);
      const Eigen::Vector3f neighbor_normal(normal_rad_fb[krow][kcol][0],
                                            normal_rad_fb[krow][kcol][1],
                                            normal_rad_fb[krow][kcol][2]);
      const float neighbor_radius = normal_rad_fb[krow][kcol][3];
      const float angle = abs(GetVectorsAngle(normal, neighbor_normal));

      const float dist = (pos - neighbor_pos).squaredNorm();
      if (dist < max_dist * max_dist && dist < radius + neighbor_radius &&
          angle <= max_angle) {
        ++count;
        if (dist < nearest_dist) {
          nearest_fb_idx = krow * width + kcol;
          nearest_dist = dist;
        }
      }
    }
  }

  if (count == 4) {
    merge_map[row][col] = nearest_fb_idx;
  }
}

__global__ void PreventDoubleMerges_gpu_kernel(
    PackedAccessor<int64_t, 1> merge_map) {
  int my_fb_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (my_fb_idx < merge_map.size(0)) return;

  const long merge_fb_idx = merge_map[my_fb_idx];
  if (merge_fb_idx >= 0) {
    long other_fb_idx = merge_map[merge_fb_idx];
    if (my_fb_idx < merge_fb_idx) {
      merge_map[my_fb_idx] = -1;
    }
  }
}

__global__ void ConvertFramebufferToSurfelIndices_gpu_kernel(
    const PackedAccessor<int32_t, 3> surfel_idx_fb,
    PackedAccessor<int64_t, 2> merge_map) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = surfel_idx_fb.size(1);
  const int height = surfel_idx_fb.size(0);

  if (row >= height || col >= width) return;

  const long merge_fb_idx = merge_map[row][col];
  if (merge_fb_idx < 0) return;

  const int merge_row = merge_fb_idx / width;
  const int merge_col = merge_fb_idx % width;

  const long merge_surfel_idx = surfel_idx_fb[merge_row][merge_col][0];
  merge_map[row][col] = merge_surfel_idx;
}
}  // namespace

void FindMergeableSurfels(const torch::Tensor &pos_fb,
                          const torch::Tensor &normal_rad_fb,
                          const torch::Tensor &idx_fb, torch::Tensor merge_map,
                          float max_dist, float max_angle, int neighbor_size) {
  const int width = pos_fb.size(1);
  const int height = pos_fb.size(0);

  const CudaKernelDims kd_img = Get2DKernelDims(width, height);

  FindMergeable_gpu_kernel<<<kd_img.grid, kd_img.block>>>(
      pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      normal_rad_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      merge_map.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
      max_dist, max_angle, neighbor_size);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  const CudaKernelDims kd_arr = Get1DKernelDims(merge_map.size(0));
  auto linear_merge_map = merge_map.view({-1});
  PreventDoubleMerges_gpu_kernel<<<kd_arr.grid, kd_arr.block>>>(
      linear_merge_map
          .packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>());
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  ConvertFramebufferToSurfelIndices_gpu_kernel<<<kd_img.grid, kd_img.block>>>(
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      merge_map
          .packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>());
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}
}  // namespace fiontb
