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
struct FindMergesKernel {
  const IndexMapAccessor<dev> model;
  typename Accessor<dev, int64_t, 2>::T merge_map;

  const float max_dist;
  const float max_angle;
  const int neighbor_size;
  const float stable_conf_thresh;

  FindMergesKernel(const IndexMap &model, const torch::Tensor &merge_map,
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

    int nearest_local_surfel_idx = -1;
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
            // nearest_surfel_idx = model.index(krow, kcol);
            nearest_local_surfel_idx = model.to_linear_index(krow, kcol);
            nearest_sq_dist = sq_dist;
          }
        }
      }
    }

    if (count >= MAX_MERGE_VIOLATIONS) {
      merge_map[row][col] = nearest_local_surfel_idx;
    }
  }
};

void PreventDoubleMerges_cpu(torch::TensorAccessor<int64_t, 1> merge_map) {
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

template <Device dev>
struct MergeKernel {
  const IndexMapAccessor<dev> indexmap;
  typename Accessor<dev, int64_t, 2>::T merge_map;
  SurfelModelAccessor<dev> model;

  MergeKernel(const IndexMap &indexmap, const torch::Tensor &merge_map,
              MappedSurfelModel model)
      : indexmap(indexmap),
        merge_map(Accessor<dev, int64_t, 2>::Get(merge_map)),
        model(model) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    const int64_t local_source_idx = merge_map[row][col];
    if (local_source_idx < 0) return;

    int64_t source_idx = indexmap.index(local_source_idx);
    merge_map[row][col] = source_idx;

    const int64_t target_idx = indexmap.index(row, col);

    const float tgt_conf = model.confidences[target_idx];
    const float src_conf = model.confidences[source_idx];
    const float conf_total = tgt_conf + src_conf;

    model.set_position(target_idx, (model.position(target_idx) * tgt_conf +
                                    model.position(source_idx) * src_conf) /
                                       conf_total);
    model.set_normal(target_idx, (model.normal(target_idx) * tgt_conf +
                                  model.normal(source_idx) * src_conf) /
                                     conf_total);
    model.set_color(target_idx, (model.color(target_idx) * tgt_conf +
                                 model.color(source_idx) * src_conf) /
                                    conf_total);
    
    for (int64_t i = 0; i < model.features.size(0); ++i) {
      const float tgt_feat_channel = model.features[i][target_idx];
      const float src_feat_channel = model.features[i][source_idx];

      model.features[i][target_idx] =
          (tgt_feat_channel * tgt_conf + src_feat_channel * src_conf) /
          conf_total;
    }

    model.confidences[target_idx] = conf_total;
    model.times[target_idx] =
        max(model.times[target_idx], model.times[source_idx]);
  }
};
}  // namespace

void SurfelFusionOp::Merge(const IndexMap &indexmap, torch::Tensor merge_map,
                           MappedSurfelModel model, float max_dist,
                           float max_angle, int neighbor_size,
                           float stable_conf_thresh) {
  const auto ref_device = indexmap.get_device();
  model.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, merge_map);

  if (ref_device.is_cuda()) {
    FindMergesKernel<kCUDA> kernel(indexmap, merge_map, max_dist, max_angle,
                                   neighbor_size, stable_conf_thresh);
    Launch2DKernelCUDA(kernel, indexmap.get_width(), indexmap.get_height());

  } else {
    FindMergesKernel<kCPU> kernel(indexmap, merge_map, max_dist, max_angle,
                                  neighbor_size, stable_conf_thresh);
    Launch2DKernelCPU(kernel, indexmap.get_width(), indexmap.get_height());
  }

  auto linear_merge_map = merge_map.view({-1}).to(torch::kCPU);
  PreventDoubleMerges_cpu(linear_merge_map.accessor<int64_t, 1>());
  merge_map.copy_(linear_merge_map.view(merge_map.sizes()), false);

  if (ref_device.is_cuda()) {
    // auto dev_merge_map =
    // linear_merge_map.view(merge_map.size()).to(torch::kCUDA);
    MergeKernel<kCUDA> kernel(indexmap, merge_map, model);
    Launch2DKernelCUDA(kernel, indexmap.get_width(), indexmap.get_height());
  } else {
    // auto dev_merge_map =
    // linear_merge_map.view(merge_map.size()).to(torch::kCPU);
    MergeKernel<kCPU> kernel(indexmap, merge_map, model);
    Launch2DKernelCPU(kernel, indexmap.get_width(), indexmap.get_height());
  }
}
}  // namespace fiontb
