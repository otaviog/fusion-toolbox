#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "cuda_error.hpp"
#include "helper_math.h"

namespace fiontb {

typedef torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
PackedUInt8Accessor1D;

typedef torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
  size_t> PackedFloatAccessor2D;
typedef torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits,
  size_t> PackedFloatAccessor3D;

typedef torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits,
  size_t> PackedInt32Accessor2D;
typedef torch::PackedTensorAccessor<int, 3, torch::RestrictPtrTraits,
  size_t> PackedInt32Accessor3D;


inline __device__ float AngleBetweenNormals(float3 norm1, float3 norm2) {
  return acos(dot(norm1, norm2) / (length(norm1) * length(norm2)));
}

inline __device__ float3 to_float3(
    const PackedFloatAccessor3D acc, int row, int col) {
  return make_float3(acc[row][col][0], acc[row][col][1], acc[row][col][2]);
}

__global__ void FindLiveToModelMerges_gpu_kernel(
    const PackedFloatAccessor3D live_pos_fb,
    const PackedFloatAccessor3D live_normal_fb,
    const PackedInt32Accessor3D live_idx_fb,
    const PackedFloatAccessor3D model_pos_fb,
    const PackedFloatAccessor3D model_normal_fb,
    const PackedInt32Accessor3D model_idx_fb,
    PackedInt32Accessor3D merge_map_fb, int scale,
    int search_size, float max_normal_angle) {
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

  const float3 ray = to_float3(live_pos_fb, row, col);
  const float lambda = sqrt(ray.x*ray.x + ray.y*ray.y + 1);

  const float3 view_normal = to_float3(live_normal_fb, row, col);

  const int xstart = max(col*scale - search_size, 0);
  const int xend = min(col*scale + search_size, int(model_pos_fb.size(1)) - 1);

  const int ystart = max(row*scale - search_size, 0);
  const int yend = min(row*scale + search_size, int(model_pos_fb.size(0)) - 1);

  float best_dist = 10000;
  int best_model = -1;

  for(int krow = ystart; krow <= yend; krow++) {
    for(int kcol = xstart; kcol <= xend; kcol++) {
      if (model_idx_fb[krow][kcol][1] == 0) continue;

      const int current = model_idx_fb[krow][kcol][0];

      const float3 vert = to_float3(model_pos_fb, krow, kcol);
      if(abs((vert.z * lambda) - (ray.z * lambda)) >= 0.05)
        continue;

      const float dist = length(cross(ray, vert)) / length(ray);
      const float3 normal = to_float3(model_normal_fb, krow, kcol);

      if(dist < best_dist
         && (abs(normal.z) < 0.75f
             || abs(AngleBetweenNormals(normal, view_normal)) < max_normal_angle)) {
          best_dist = dist;
          best_model = current;
      }
    }
  }

  merge_map_fb[row][col][1] = best_model;
}

__global__ void FindFeatLiveToModelMerges_gpu_kernel(
    const PackedFloatAccessor3D live_pos_fb,
    const PackedFloatAccessor3D live_normal_fb,
    const PackedInt32Accessor3D live_idx_fb,
    const PackedFloatAccessor2D live_feats,
    const PackedFloatAccessor3D model_pos_fb,
    const PackedFloatAccessor3D model_normal_fb,
    const PackedInt32Accessor3D model_idx_fb,
    const PackedFloatAccessor2D model_feats,
    PackedInt32Accessor3D merge_map_fb, int scale,
    int search_size, float max_normal_angle) {
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

  const float3 ray = to_float3(live_pos_fb, row, col);
  const float lambda = sqrt(ray.x*ray.x + ray.y*ray.y + 1);

  const float3 view_normal = to_float3(live_normal_fb, row, col);

  const int xstart = max(col*scale - search_size, 0);
  const int xend = min(col*scale + search_size, int(model_pos_fb.size(1)) - 1);

  const int ystart = max(row*scale - search_size, 0);
  const int yend = min(row*scale + search_size, int(model_pos_fb.size(0)) - 1);

  float best_dist = 10000;
  int best_model = -1;

  for(int krow = ystart; krow <= yend; krow++) {
    for(int kcol = xstart; kcol <= xend; kcol++) {
      if (model_idx_fb[krow][kcol][1] == 0) continue;

      const int current = model_idx_fb[krow][kcol][0];

      float sqr_feat_dist = 0.0f;
      for (size_t i=0; i<model_feats.size(1); ++i) {        
        const float diff = model_feats[current][i] - live_feats[live_idx][i];
        sqr_feat_dist += diff*diff;
      }

      if (sqr_feat_dist < 3.0f) continue;
      
      const float3 vert = to_float3(model_pos_fb, krow, kcol);
      if(abs((vert.z * lambda) - (ray.z * lambda)) >= 0.05)
        continue;

      //const float dist = length(cross(ray, vert)) / length(ray);
      const float dist = sqr_feat_dist;
      if(dist >= best_dist) continue;
      
      const float3 normal = to_float3(model_normal_fb, krow, kcol);
      //if (abs(normal.z) < 0.75f
      //|| abs(AngleBetweenNormals(normal, view_normal)) <
      //max_normal_angle) {
      if (true) {
          best_dist = dist;
          best_model = current;
      }
    }
  }

  merge_map_fb[row][col][1] = best_model;
}

torch::Tensor FindLiveToModelMerges(
    const torch::Tensor &live_pos_fb,
    const torch::Tensor &live_normal_fb,
    const torch::Tensor &live_idx_fb,
    const torch::Tensor &live_feats,
    const torch::Tensor &model_pos_fb,
    const torch::Tensor &model_normal_fb,
    const torch::Tensor &model_idx_fb,
    const torch::Tensor &model_feats,
    float max_normal_angle, bool use_feats) {
  const int width = live_pos_fb.size(1);
  const int height = live_pos_fb.size(0);

  torch::Tensor merge_map =
      torch::empty({height, width, 3},
                   torch::TensorOptions(torch::kInt32).device(live_pos_fb.device()));

  const float scale = model_pos_fb.size(0)/height;
  const float window_multiplier = 2;
  const int search_size = int(scale*window_multiplier);

  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);
  if (!use_feats) {
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
  } else {
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
  }
  
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());

  return merge_map;

}

__global__ void CarveSpace_gpu_kernel(const PackedFloatAccessor3D stable_pos_fb,
                                      const PackedInt32Accessor3D stable_idx_fb,
                                      const PackedFloatAccessor3D view_pos_fb,
                                      const PackedInt32Accessor3D view_idx_fb,
                                      PackedUInt8Accessor1D free_mask,
                                      int neighbor_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = view_pos_fb.size(1);
  const int height = view_pos_fb.size(0);

  if (row >= height || col >= width) return;
  if (view_idx_fb[row][col][1] == 0) return;

  int violantion_count = 0;

  const float view_pos_z = view_pos_fb[row][col][2];
  for (int krow = max(0, row - neighbor_size);
       krow < min(height - 1, row + neighbor_size); ++krow) {
    for (int kcol = max(0, col - neighbor_size);
         kcol < min(width - 1, col + neighbor_size); ++kcol) {
      const int stable_flag = stable_idx_fb[krow][kcol][1];

      if (stable_flag == 0) {
        continue;
      }

      const int stable_idx = stable_idx_fb[krow][kcol][0];
      const float stable_z = stable_pos_fb[krow][kcol][2];
      if ((stable_z - view_pos_z) > 0.1) {
        ++violantion_count;
      }
    }
  }

  if (violantion_count > 1) {
    const int view_idx = view_idx_fb[row][col][0];
    free_mask[view_idx] = 1;
  }
}

void CarveSpace(const torch::Tensor stable_pos_fb,
                const torch::Tensor stable_idx_fb,
                const torch::Tensor view_pos_fb,
                const torch::Tensor view_idx_fb, torch::Tensor free_mask,
                int neighbor_size) {
  const int width = view_pos_fb.size(1);
  const int height = view_pos_fb.size(0);

  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);

  CarveSpace_gpu_kernel<<<grid_size, block_dim>>>(
      stable_pos_fb
      .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      stable_idx_fb
          .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      view_pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      view_idx_fb
          .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      free_mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      neighbor_size);
  CudaCheck();
}

__global__ void FindMergeable_gpu_kernel(
    const PackedFloatAccessor3D pos_fb,
    const PackedFloatAccessor3D normal_rad_fb,
    const PackedInt32Accessor3D idx_fb, PackedInt32Accessor2D merge_map,
    float max_dist, float max_angle, int neighbor_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = pos_fb.size(1);
  const int height = pos_fb.size(0);

  if (row >= height || col >= width) return;

  merge_map[row][col] = -1;
  if (idx_fb[row][col][1] == 0) return;

  const float3 pos = make_float3(pos_fb[row][col][0], pos_fb[row][col][1],
                                 pos_fb[row][col][2]);
  const float3 normal =
      make_float3(normal_rad_fb[row][col][0], normal_rad_fb[row][col][1],
                  normal_rad_fb[row][col][2]);
  const float radius = normal_rad_fb[row][col][3];

  int best_local_idx = -1;
  float best_dist = 999999.0f;
  for (int krow = max(0, row - neighbor_size);
       krow < min(height - 1, row + neighbor_size); ++krow) {
    for (int kcol = max(0, col - neighbor_size);
         kcol < min(width - 1, col + neighbor_size); ++kcol) {
      if (idx_fb[krow][kcol][1] == 0) continue;

      const float3 neighbor_pos = make_float3(
          pos_fb[krow][kcol][0], pos_fb[krow][kcol][1], pos_fb[krow][kcol][2]);
      const float3 neighbor_normal = make_float3(normal_rad_fb[krow][kcol][0],
                                                 normal_rad_fb[krow][kcol][1],
                                                 normal_rad_fb[krow][kcol][2]);
      const float neighbor_radius = normal_rad_fb[krow][kcol][3];
      const float angle = abs(acos(dot(normal, neighbor_normal)));

      const float dist = length(pos - neighbor_pos);

      if (dist <= max_dist && angle <= max_angle &&
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

__global__ void Merge_gpu_kernel(
    const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits,
                                      size_t>
        merge_map,
    const torch::PackedTensorAccessor<int32_t, 3, torch::RestrictPtrTraits,
                                      size_t>
        idx_fb,
    torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
        free_mask) {
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

void MergeRedundant(const torch::Tensor &pos_fb,
                    const torch::Tensor &normal_rad_fb,
                    const torch::Tensor &idx_fb, torch::Tensor free_mask,
                    float max_dist, float max_angle, int neighbor_size) {
  const int width = pos_fb.size(1);
  const int height = pos_fb.size(0);

  const dim3 block_dim = dim3(16, 16);
  const dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);

  torch::Tensor merge_map =
      torch::empty({height, width},
                   torch::TensorOptions(torch::kInt32).device(torch::kCUDA, 0));

  FindMergeable_gpu_kernel<<<grid_size, block_dim>>>(
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

  const int block_size = 128;
  const int num_blocks = merge_map.size(0) / block_size + 1;

  Merge_gpu_kernel<<<num_blocks, block_size>>>(
      merge_map.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),
      idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      free_mask
          .packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>());
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb
