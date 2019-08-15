#include <limits>

#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

const int MAX_MERGE_VIOLATIONS = 1;

template <typename FloatAccessor3D, typename Int32Accessor3D>
struct Framebuffer {
  Framebuffer(const FloatAccessor3D pos_fb, const FloatAccessor3D normal_rad_fb,
              const Int32Accessor3D idx_fb)
      : position_conf(pos_fb), normal_radius(normal_rad_fb), index(idx_fb) {}

  __device__ __host__ int width() const { return position_conf.size(1); }
  __device__ __host__ int height() const { return position_conf.size(0); }
  __device__ __host__ bool empty(int row, int col) const {
    return index[row][col][1] == 0;
  }

  const FloatAccessor3D position_conf;
  const FloatAccessor3D normal_radius;
  const Int32Accessor3D index;
};

typedef Framebuffer<PackedAccessor<float, 3>, PackedAccessor<int32_t, 3>>
    CUDAFramebuffer;

typedef Framebuffer<torch::TensorAccessor<float, 3>,
                    torch::TensorAccessor<int32_t, 3>>
    CPUFramebuffer;

__global__ void FindMergeable_gpu_kernel(CUDAFramebuffer model,
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
  float nearest_dist = CUDART_INF_F;
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
      if (dist <= max_dist * max_dist && dist < radius + neighbor_radius &&
          angle <= max_angle) {
        ++count;
        if (dist < nearest_dist) {
          nearest_fb_idx = krow * model.width() + kcol;
          nearest_dist = dist;
        }
      }
    }
  }

  if (count >= MAX_MERGE_VIOLATIONS) {
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

/**
 * CPU Kernels
 */
void FindMergeable_cpu_kernel(CPUFramebuffer model,
                              torch::TensorAccessor<int64_t, 2> merge_map,
                              float max_dist, float max_angle,
                              int neighbor_size, float stable_conf_thresh) {
  for (int row = 0; row < model.height(); ++row) {
    for (int col = 0; col < model.width(); ++col) {
      merge_map[row][col] = -1;
      if (model.empty(row, col)) continue;
      if (model.position_conf[row][col][3] < stable_conf_thresh) continue;

      const Eigen::Vector3f pos(to_vec3<float>(model.position_conf[row][col]));
      const Eigen::Vector3f normal(
          to_vec3<float>(model.normal_radius[row][col]));
      const float radius = model.normal_radius[row][col][3];

      int nearest_fb_idx = -1;
      float nearest_sq_dist = std::numeric_limits<float>::infinity();
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

          const float sq_dist = (pos - neighbor_pos).squaredNorm();
          if (sq_dist <= max_dist * max_dist && sq_dist < radius + neighbor_radius &&
              angle <= max_angle) {
            ++count;
            if (sq_dist < nearest_sq_dist) {
              nearest_fb_idx = krow * model.width() + kcol;
              nearest_sq_dist = sq_dist;
            }
          }
        }
      }

      if (count >= MAX_MERGE_VIOLATIONS) {
        merge_map[row][col] = nearest_fb_idx;
      }
    }
  }
}

void PreventDoubleMerges_cpu_kernel(
    torch::TensorAccessor<int64_t, 1> merge_map) {
  /**
   *   Cases: 
   *   i -> -1        # i merges with no one
   *   i ->  j ->  k (i) # i merges with j, and j merges with i
   *   i ->  j -> -1  # i whats to merge with j
   *   i ->  j ->  k  # i merges with j, and j merges with k
   */
  for (int i = 0; i < merge_map.size(0); ++i) {
    const int64_t j = merge_map[i];
    if (j >= 0) {
      const int64_t k = merge_map[j];
	  if (i == k) {
		merge_map[j] = -1;
	  } else {
		merge_map[i] = -1;
	  }
	  
	  //if (merge_map[other_idx] == -1) {
		// Someone already merged with this one
		//merge_map[i] = -1;
	  //}
    }
  }
}

void ConvertFramebufferToSurfelIndices_cpu_kernel(
    const torch::TensorAccessor<int32_t, 3> surfel_idx_fb,
    torch::TensorAccessor<int64_t, 2> merge_map) {
  const int width = surfel_idx_fb.size(1);
  const int height = surfel_idx_fb.size(0);

  for (int row = 0; row < surfel_idx_fb.size(0); ++row) {
    for (int col = 0; col < surfel_idx_fb.size(1); ++col) {
      if (row >= height || col >= width) continue;

      const long merge_fb_idx = merge_map[row][col];
      if (merge_fb_idx < 0) continue;

      const int merge_row = merge_fb_idx / width;
      const int merge_col = merge_fb_idx % width;

      const long merge_surfel_idx = surfel_idx_fb[merge_row][merge_col][0];
      merge_map[row][col] = merge_surfel_idx;
    }
  }
}
}  // namespace

void FindMergeableSurfels(const torch::Tensor &pos_fb,
                          const torch::Tensor &normal_rad_fb,
                          const torch::Tensor &idx_fb, torch::Tensor merge_map,
                          float max_dist, float max_angle, int neighbor_size,
                          float stable_conf_thresh) {
  if (pos_fb.is_cuda()) {
    FTB_CHECK(normal_rad_fb.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(idx_fb.is_cuda(), "Expected a cuda tensor");

    CUDAFramebuffer model(
        pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        normal_rad_fb
            .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        idx_fb.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>());

    const CudaKernelDims kd_img =
        Get2DKernelDims(model.width(), model.height());

    FindMergeable_gpu_kernel<<<kd_img.grid, kd_img.block>>>(
        model,
        merge_map
            .packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
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
  } else {
    CPUFramebuffer model(pos_fb.accessor<float, 3>(),
                         normal_rad_fb.accessor<float, 3>(),
                         idx_fb.accessor<int32_t, 3>());

    FindMergeable_cpu_kernel(model, merge_map.accessor<int64_t, 2>(), max_dist,
                             max_angle, neighbor_size, stable_conf_thresh);

    auto linear_merge_map = merge_map.view({-1});
    PreventDoubleMerges_cpu_kernel(linear_merge_map.accessor<int64_t, 1>());

    ConvertFramebufferToSurfelIndices_cpu_kernel(
        idx_fb.accessor<int32_t, 3>(), merge_map.accessor<int64_t, 2>());
  }
}
}  // namespace fiontb