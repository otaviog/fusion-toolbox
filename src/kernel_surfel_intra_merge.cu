#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "math.hpp"

namespace fiontb {
namespace {
struct Framebuffer {
  Framebuffer(const PackedAccessor<float, 3> pos_fb,
              const PackedAccessor<float, 3> normal_rad_fb,
              const PackedAccessor<int32_t, 3> idx_fb)
      : position_conf(pos_fb), normal_radius(normal_rad_fb), index(idx_fb) {}

  __device__ __host__ int width() const { return position_conf.size(1); }
  __device__ __host__ int height() const { return position_conf.size(0); }
  __device__ bool empty(int row, int col) const {
    return index[row][col][1] == 0;
  }

  const PackedAccessor<float, 3> position_conf;
  const PackedAccessor<float, 3> normal_radius;
  const PackedAccessor<int32_t, 3> index;
};

__global__ void FindMergeable_gpu_kernel(Framebuffer model,
                                         PackedAccessor<int64_t, 2> merge_map,
                                         float max_dist, float max_angle,
                                         int neighbor_size,
                                         float stable_conf_thresh) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= model.height() || col >= model.width()) return;

  merge_map[row][col] = -1;
  if (model.empty(row, col)) return;
  if (model.position_conf[row][col][3] < stable_conf_thresh) return;

  const Eigen::Vector3f pos(to_vec3<float>(model.position_conf[row][col]));
  const Eigen::Vector3f normal(to_vec3<float>(model.normal_radius[row][col]));
  const float radius = model.normal_radius[row][col][3];

  int nearest_fb_idx = -1;
  float nearest_dist = max_dist * max_dist;
  int count = 0;

  const int start_row = max(row - neighbor_size, 0);
  const int end_row = min(row + neighbor_size, model.height() - 1);

  const int start_col = max(col - neighbor_size, 0);
  const int end_col = min(col + neighbor_size, model.width() - 1);

  for (int krow = start_row; krow <= end_row; ++krow) {
    for (int kcol = start_col; kcol <= end_col; ++kcol) {
      if (krow == row && kcol == col) continue;
      if (model.empty(krow, kcol)) continue;
      if (model.position_conf[krow][kcol][3] < stable_conf_thresh) continue;

      const Eigen::Vector3f neighbor_pos(
          to_vec3<float>(model.position_conf[krow][kcol]));

      const Eigen::Vector3f neighbor_normal(
          to_vec3<float>(model.normal_radius[krow][kcol]));
      const float neighbor_radius = model.normal_radius[krow][kcol][3];
      const float angle = abs(GetVectorsAngle(normal, neighbor_normal));

      const float dist = (pos - neighbor_pos).squaredNorm();
      if (dist < max_dist * max_dist && dist < radius + neighbor_radius &&
          angle <= max_angle) {
        ++count;
        if (dist < nearest_dist) {
          nearest_fb_idx = krow * model.width() + kcol;
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
                          float max_dist, float max_angle, int neighbor_size,
                          float stable_conf_thresh) {
  Framebuffer model(
      pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      normal_rad_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>());

  const CudaKernelDims kd_img = Get2DKernelDims(model.width(), model.height());

  FindMergeable_gpu_kernel<<<kd_img.grid, kd_img.block>>>(
      model,
      merge_map.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
      max_dist, max_angle, neighbor_size, stable_conf_thresh);
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
