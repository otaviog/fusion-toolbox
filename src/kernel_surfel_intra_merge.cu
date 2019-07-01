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
    PackedAccessor<int32_t, 2> merge_map, float max_dist, float max_angle,
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

  int best_local_idx = -1;
  float best_dist = 999999.0f;
  for (int krow = max(0, row - neighbor_size);
       krow < min(height - 1, row + neighbor_size); ++krow) {
    for (int kcol = max(0, col - neighbor_size);
         kcol < min(width - 1, col + neighbor_size); ++kcol) {
      if (idx_fb[krow][kcol][1] == 0) continue;

      const Eigen::Vector3f neighbor_pos(
          pos_fb[krow][kcol][0], pos_fb[krow][kcol][1], pos_fb[krow][kcol][2]);
      const Eigen::Vector3f neighbor_normal(normal_rad_fb[krow][kcol][0],
                                            normal_rad_fb[krow][kcol][1],
                                            normal_rad_fb[krow][kcol][2]);
      const float neighbor_radius = normal_rad_fb[krow][kcol][3];
      const float angle = abs(GetVectorsAngle(normal, neighbor_normal));

      const float dist = (pos - neighbor_pos).squaredNorm();
      if (dist <= max_dist * max_dist && angle <= max_angle &&
          dist < radius + neighbor_radius) {
        best_local_idx = krow * width + kcol;
        best_dist = dist;
      }
    }
  }

  merge_map[row][col] = best_local_idx;
}

void PreventDoubleMerge(torch::TensorAccessor<int32_t, 1> merge_map) {
  for (long i = 0; i < merge_map.size(0); ++i) {
    const int merge_idx = merge_map[i];
    if (merge_idx >= 0) {
      merge_map[merge_idx] = -1;
    }
  }
}

__global__ void Merge_gpu_kernel(const PackedAccessor<int32_t, 1> merge_map,
                                 const PackedAccessor<int32_t, 3> idx_fb,
                                 PackedAccessor<uint8_t, 1> free_mask) {
  int tensor_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tensor_idx < merge_map.size(0)) {
    const int merge_local_idx = merge_map[tensor_idx];
    if (merge_local_idx >= 0) {
      const int row = merge_local_idx / idx_fb.size(1);
      const int col = merge_local_idx % idx_fb.size(1);
      const int merge_idx = idx_fb[row][col][0];
      free_mask[merge_idx] = 1;
    }
  }
}
}  // namespace

void MergeRedundant(const torch::Tensor &pos_fb,
                    const torch::Tensor &normal_rad_fb,
                    const torch::Tensor &idx_fb, torch::Tensor free_mask,
                    float max_dist, float max_angle, int neighbor_size) {
  const int width = pos_fb.size(1);
  const int height = pos_fb.size(0);

  const CudaKernelDims kd_img = Get2DKernelDims(width, height);

  torch::Tensor merge_map =
      torch::empty({height, width},
                   torch::TensorOptions(torch::kInt32).device(torch::kCUDA, 0));

  FindMergeable_gpu_kernel<<<kd_img.grid, kd_img.block>>>(
      pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      normal_rad_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      merge_map.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
      max_dist, max_angle, neighbor_size);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  torch::Tensor merge_map_cpu = merge_map.cpu().view({-1});
  PreventDoubleMerge(merge_map_cpu.accessor<int32_t, 1>());

  merge_map = merge_map_cpu.to(merge_map.device());

  const CudaKernelDims kd_arr = Get1DKernelDims(merge_map.size(0));
  Merge_gpu_kernel<<<kd_arr.grid, kd_arr.block>>>(
      merge_map.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      free_mask
          .packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>());
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}
}  // namespace fiontb
