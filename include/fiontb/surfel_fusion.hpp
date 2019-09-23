#pragma once

#include <torch/torch.h>

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
}  // namespace fiontb
