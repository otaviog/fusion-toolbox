#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "math.hpp"

namespace fiontb {
static __global__ void FindLiveToModelMerges_gpu_kernel(
    const PackedAccessor<float, 3> live_pos_fb,
    const PackedAccessor<float, 3> live_normal_fb,
    const PackedAccessor<int32_t, 3> live_idx_fb,
    const PackedAccessor<float, 3> model_pos_fb,
    const PackedAccessor<float, 3> model_normal_fb,
    const PackedAccessor<int32_t, 3> model_idx_fb,
    PackedAccessor<int32_t, 3> merge_map_fb, int scale, int search_size,
    float max_normal_angle) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int live_fb_width = live_pos_fb.size(1);
  const int live_fb_height = live_pos_fb.size(0);

  if (row >= live_pos_fb.size(0) || col >= live_pos_fb.size(1)) return;
  merge_map_fb[row][col][0] = 0;
  merge_map_fb[row][col][1] = -1;
  merge_map_fb[row][col][2] = -1;

  if (live_idx_fb[row][col][1] == 0) return;

  merge_map_fb[row][col][0] = 1;
  merge_map_fb[row][col][2] = live_idx_fb[row][col][0];

  const Eigen::Vector3f ray(live_pos_fb[row][col][0], live_pos_fb[row][col][1],
                            live_pos_fb[row][col][2]);
  const float lambda = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1);

  const Eigen::Vector3f view_normal(live_normal_fb[row][col][0],
                                    live_normal_fb[row][col][1],
                                    live_normal_fb[row][col][2]);

  const int xstart = max(col * scale - search_size, 0);
  const int xend =
      min(col * scale + search_size, int(model_pos_fb.size(1)) - 1);

  const int ystart = max(row * scale - search_size, 0);
  const int yend =
      min(row * scale + search_size, int(model_pos_fb.size(0)) - 1);

  float best_dist = 10000;
  int best_model = -1;

  for (int krow = ystart; krow <= yend; krow++) {
    for (int kcol = xstart; kcol <= xend; kcol++) {
      if (model_idx_fb[krow][kcol][1] == 0) continue;

      const int current = model_idx_fb[krow][kcol][0];

      const Eigen::Vector3f vert(to_vec3<float>(model_pos_fb[krow][kcol]));
      if (abs((vert[2] * lambda) - (ray[2] * lambda)) >= 0.05) continue;

      const float dist = ray.cross(vert).norm() / ray.norm();
      const Eigen::Vector3f normal(to_vec3<float>(model_normal_fb[krow][kcol]));

      if (dist < best_dist &&
          (abs(normal[2]) < 0.75f ||
           abs(GetVectorsAngle(normal, view_normal)) < max_normal_angle)) {
        best_dist = dist;
        best_model = current;
      }
    }
  }

  merge_map_fb[row][col][1] = best_model;
}

torch::Tensor FindLiveToModelMerges(const torch::Tensor &live_pos_fb,
                                    const torch::Tensor &live_normal_fb,
                                    const torch::Tensor &live_idx_fb,
                                    const torch::Tensor &model_pos_fb,
                                    const torch::Tensor &model_normal_fb,
                                    const torch::Tensor &model_idx_fb,
                                    float max_normal_angle) {
  const int width = live_pos_fb.size(1);
  const int height = live_pos_fb.size(0);

  torch::Tensor merge_map = torch::empty(
      {height, width, 3},
      torch::TensorOptions(torch::kInt32).device(live_pos_fb.device()));

  const float scale = model_pos_fb.size(0) / height;
  const float window_multiplier = 2;
  const int search_size = int(scale * window_multiplier);

  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);
  FindLiveToModelMerges_gpu_kernel<<<grid_size, block_dim>>>(
      live_pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      live_normal_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      live_idx_fb.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      model_pos_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      model_normal_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      model_idx_fb.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      merge_map.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      int(scale), search_size, max_normal_angle);

  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  return merge_map;
}

static __global__ void FindFeatLiveToModelMerges_gpu_kernel(
    const PackedAccessor<float, 3> live_pos_fb,
    const PackedAccessor<float, 3> live_normal_fb,
    const PackedAccessor<int32_t, 3> live_idx_fb,
    const PackedAccessor<float, 2> live_feats,
    const PackedAccessor<float, 3> model_pos_fb,
    const PackedAccessor<float, 3> model_normal_fb,
    const PackedAccessor<int32_t, 3> model_idx_fb,
    const PackedAccessor<float, 2> model_feats,
    PackedAccessor<int32_t, 3> merge_map_fb, int scale, int search_size,
    float max_normal_angle) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int live_fb_width = live_pos_fb.size(1);
  const int live_fb_height = live_pos_fb.size(0);

  if (row >= live_pos_fb.size(0) || col >= live_pos_fb.size(1)) return;
  merge_map_fb[row][col][0] = 0;
  merge_map_fb[row][col][1] = -1;
  merge_map_fb[row][col][2] = -1;

  if (live_idx_fb[row][col][1] == 0) return;
  const int live_idx = live_idx_fb[row][col][0];
  merge_map_fb[row][col][0] = 1;
  merge_map_fb[row][col][2] = live_idx;

  const Eigen::Vector3f ray(to_vec3<float>(live_pos_fb[row][col]));
  const float lambda = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1);

  const Eigen::Vector3f view_normal(to_vec3<float>(live_normal_fb[row][col]));

  const int xstart = max(col * scale - search_size, 0);
  const int xend =
      min(col * scale + search_size, int(model_pos_fb.size(1)) - 1);

  const int ystart = max(row * scale - search_size, 0);
  const int yend =
      min(row * scale + search_size, int(model_pos_fb.size(0)) - 1);

  float best_dist = 10000;
  int best_model = -1;

  for (int krow = ystart; krow <= yend; krow++) {
    for (int kcol = xstart; kcol <= xend; kcol++) {
      if (model_idx_fb[krow][kcol][1] == 0) continue;

      const int current = model_idx_fb[krow][kcol][0];

      float sqr_feat_dist = 0.0f;
      for (size_t i = 0; i < model_feats.size(1); ++i) {
        const float diff = model_feats[current][i] - live_feats[live_idx][i];
        sqr_feat_dist += diff * diff;
      }

      if (sqr_feat_dist < 3.0f) continue;

      const Eigen::Vector3f vert(model_pos_fb[krow][kcol][0],
                                 model_pos_fb[krow][kcol][1],
                                 model_pos_fb[krow][kcol][2]);
      if (abs((vert[2] * lambda) - (ray[2] * lambda)) >= 0.05) continue;

      // const float dist = length(cross(ray, vert)) / length(ray);
      const float dist = sqr_feat_dist;
      if (dist >= best_dist) continue;

      const Eigen::Vector3f normal(model_normal_fb[krow][kcol][0],
                                   model_normal_fb[krow][kcol][1],
                                   model_normal_fb[krow][kcol][2]);
      // if (abs(normal.z) < 0.75f
      //|| abs(AngleBetweenNormals(normal, view_normal)) <
      // max_normal_angle) {
      if (true) {
        best_dist = dist;
        best_model = current;
      }
    }
  }

  merge_map_fb[row][col][1] = best_model;
}

torch::Tensor FindFeatLiveToModelMerges(
    const torch::Tensor &live_pos_fb, const torch::Tensor &live_normal_fb,
    const torch::Tensor &live_idx_fb, const torch::Tensor &live_feats,
    const torch::Tensor &model_pos_fb, const torch::Tensor &model_normal_fb,
    const torch::Tensor &model_idx_fb, const torch::Tensor &model_feats,
    float max_normal_angle) {
  const int width = live_pos_fb.size(1);
  const int height = live_pos_fb.size(0);

  torch::Tensor merge_map = torch::empty(
      {height, width, 3},
      torch::TensorOptions(torch::kInt32).device(live_pos_fb.device()));

  const float scale = model_pos_fb.size(0) / height;
  const float window_multiplier = 2;
  const int search_size = int(scale * window_multiplier);

  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);

  FindFeatLiveToModelMerges_gpu_kernel<<<grid_size, block_dim>>>(
      live_pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      live_normal_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      live_idx_fb.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      live_feats.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      model_pos_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      model_normal_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      model_idx_fb.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      model_feats.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      merge_map.packed_accessor<int, 3, torch::RestrictPtrTraits, size_t>(),
      int(scale), search_size, max_normal_angle);

  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  return merge_map;
}
}  // namespace fiontb