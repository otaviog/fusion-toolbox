#include "surfel_fusion_common.hpp"

#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

const int MAX_MERGE_VIOLATIONS = 1;

template <Device dev>
struct MergeKernel {
  const IndexMapAccessor<dev> model;
  typename Accessor<dev, int64_t, 2>::T merge_map;

  const float max_dist;
  const float max_angle;
  const int neighbor_size;
  const float stable_conf_thresh;

  MergeKernel(const IndexMap &model, const torch::Tensor &merge_map,
              float max_dist, float max_angle, int neighbor_size,
              float stable_conf_thresh)
      : model(IndexMapAccessor<dev>(model)),
        merge_map(Accessor<dev, int64_t, 2>::Get(merge_map)),
        max_dist(max_dist),
        max_angle(max_angle),
        neighbor_size(neighbor_size),
        stable_conf_thresh(stable_conf_thresh) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    merge_map[row][col] = -1;
    if (model.empty(row, col)) return;
    if (model.confidence(row, col) < stable_conf_thresh) return;

    const Eigen::Vector3f pos = model.position(row, col);
    const Eigen::Vector3f normal = model.normal(row, col);
    const float radius = model.radius(row, col);

    int nearest_fb_idx = -1;
    float nearest_sq_dist = NumericLimits<dev, float>::infinity();
    int count = 0;

    const int start_row = max(row - neighbor_size, 0);
    const int end_row = min(row + neighbor_size, model.height() - 1);

    const int start_col = max(col - neighbor_size, 0);
    const int end_col = min(col + neighbor_size, model.width() - 1);

    for (int krow = start_row; krow <= end_row; ++krow) {
      for (int kcol = start_col; kcol <= end_col; ++kcol) {
        if (krow == row && kcol == col) continue;
        if (model.empty(krow, kcol)) continue;
        if (model.confidence(krow, kcol) < stable_conf_thresh) continue;

        const Eigen::Vector3f neighbor_pos = model.position(krow, kcol);

        const Eigen::Vector3f neighbor_normal = model.normal(krow, kcol);
        const float neighbor_radius = model.radius(krow, kcol);
        const float angle = abs(GetVectorsAngle(normal, neighbor_normal));

        const float sq_dist = (pos - neighbor_pos).squaredNorm();
        const float dist_to = radius + neighbor_radius;
        if (sq_dist <= max_dist * max_dist && sq_dist < dist_to * dist_to &&
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
};

template <Device dev>
struct PreventDoubleMergesKernel {
  typename Accessor<dev, int64_t, 1>::T merge_map;

  PreventDoubleMergesKernel(const torch::Tensor &merge_map)
      : merge_map(Accessor<dev, int64_t, 1>::Get(merge_map)) {}

  FTB_DEVICE_HOST void operator()(int i) {
    /**
     *   Cases:
     *   i -> j (-1)      # i merges with no one
     *   i -> j ->  k (i) # i merges with j, and j merges with i
     *   i -> j -> -1     # i whats to merge with j
     *   i -> j ->  k     # i merges with j, and j merges with k
     */
    const int64_t j = merge_map[i];
    if (j >= 0) {
      const int64_t k = merge_map[j];
      if (i == k) {
        if (i < j) merge_map[j] = -1;
      } else {
        merge_map[i] = -1;
      }
    }
  }
};

template <Device dev>
struct IndexMapToSurfelIndexKernel {
  const typename Accessor<dev, int32_t, 3>::T surfel_idx_fb;
  typename Accessor<dev, int64_t, 2>::T merge_map;

  IndexMapToSurfelIndexKernel(const torch::Tensor &surfel_idx_fb,
                              torch::Tensor merge_map)
      : surfel_idx_fb(Accessor<dev, int32_t, 3>::Get(surfel_idx_fb)),
        merge_map(Accessor<dev, int64_t, 2>::Get(merge_map)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    const int width = surfel_idx_fb.size(1);
    const int height = surfel_idx_fb.size(0);

    const long merge_fb_idx = merge_map[row][col];
    if (merge_fb_idx < 0) return;

    const int merge_row = merge_fb_idx / width;
    const int merge_col = merge_fb_idx % width;

    const long merge_surfel_idx = surfel_idx_fb[merge_row][merge_col][0];
    merge_map[row][col] = merge_surfel_idx;
  }
};

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

      // if (merge_map[other_idx] == -1) {
      // Someone already merged with this one
      // merge_map[i] = -1;
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

void SurfelFusionOp::Merge(const IndexMap &model, torch::Tensor merge_map,
                           float max_dist, float max_angle, int neighbor_size,
                           float stable_conf_thresh) {
  const auto ref_device = model.get_device();
  model.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, merge_map);

  if (ref_device.is_cuda()) {
    {
      MergeKernel<kCUDA> kernel(model, merge_map, max_dist, max_angle,
                                neighbor_size, stable_conf_thresh);
      Launch2DKernelCUDA(kernel, model.get_width(), model.get_height());
    }

    auto linear_merge_map = merge_map.view({-1});

    {
      PreventDoubleMergesKernel<kCUDA> kernel(linear_merge_map);
      Launch1DKernelCUDA(kernel, linear_merge_map.size(0));
    }

    {
      IndexMapToSurfelIndexKernel<kCUDA> kernel(model.indexmap, merge_map);
      Launch2DKernelCUDA(kernel, model.get_width(), model.get_height());
    }
  } else {
    {
      MergeKernel<kCPU> kernel(model, merge_map, max_dist, max_angle,
                               neighbor_size, stable_conf_thresh);
      Launch2DKernelCPU(kernel, model.get_width(), model.get_height());
    }

    auto linear_merge_map = merge_map.view({-1});
    PreventDoubleMerges_cpu_kernel(linear_merge_map.accessor<int64_t, 1>());

    ConvertFramebufferToSurfelIndices_cpu_kernel(
        model.indexmap.accessor<int32_t, 3>(),
        merge_map.accessor<int64_t, 2>());
  }
}
}  // namespace fiontb