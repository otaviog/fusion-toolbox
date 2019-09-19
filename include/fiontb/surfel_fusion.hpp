#pragma once

#include <torch/torch.h>

#include "error.hpp"

namespace fiontb {

void CarveSpace(const torch::Tensor model_pos_fb,
                const torch::Tensor model_idx_fb, torch::Tensor mask,
                int curr_time, float stable_conf_thresh, int search_size,
                float min_z_diff);

void FindMergeableSurfels(const torch::Tensor &pos_fb,
                          const torch::Tensor &normal_rad_fb,
                          const torch::Tensor &idx_fb, torch::Tensor merge_map,
                          float max_dist, float max_angle, int neighbor_size,
                          float stable_conf_thresh);

torch::Tensor FindLiveToModelMerges(const torch::Tensor &live_pos_fb,
                                    const torch::Tensor &live_normal_fb,
                                    const torch::Tensor &live_idx_fb,
                                    const torch::Tensor &model_pos_fb,
                                    const torch::Tensor &model_normal_fb,
                                    const torch::Tensor &model_idx_fb,
                                    float max_normal_angle, int search_size);

torch::Tensor FindFeatLiveToModelMerges(
    const torch::Tensor &live_pos_fb, const torch::Tensor &live_normal_fb,
    const torch::Tensor &live_idx_fb, const torch::Tensor &live_feats,
    const torch::Tensor &model_pos_fb, const torch::Tensor &model_normal_fb,
    const torch::Tensor &model_idx_fb, const torch::Tensor &model_feats,
    float max_normal_angle, int search_size);

struct IndexMapParams {
  torch::Tensor position_confidence, normal_radius, color, indexmap;

  bool IsCuda() const { return position_confidence.is_cuda(); }

  void CheckCuda() const {
    FTB_CHECK(normal_radius.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(color.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(indexmap.is_cuda(), "Expected a cuda tensor");
  }

  void CheckCpu() const {
    FTB_CHECK(!normal_radius.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!color.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!indexmap.is_cuda(), "Expected a cpu tensor");
  }
};

struct SurfelModelParams {
  torch::Tensor positions, confidences, normals, radii, colors;

  void CheckCuda() const {
    FTB_CHECK(positions.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(confidences.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(normals.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(radii.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(colors.is_cuda(), "Expected a cuda tensor");
  }

  void CheckCpu() const {
    FTB_CHECK(!positions.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!confidences.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!normals.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!radii.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!colors.is_cuda(), "Expected a cpu tensor");
  }
};

struct FeatSurfel {
  static void MergeLive(const IndexMapParams &target_indexmap_params,
                        const IndexMapParams &live_indexmap_params,
                        const SurfelModelParams &model_params, int search_size,
                        float max_normal_angle);
};

}  // namespace fiontb
