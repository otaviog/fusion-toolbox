#pragma once

#include <torch/torch.h>

namespace fiontb {
void CarveSpace(const torch::Tensor stable_pos_fb,
                const torch::Tensor stable_idx_fb,
                const torch::Tensor view_pos_fb,
                const torch::Tensor view_idx_fb, torch::Tensor mask,
                int neighbor_size);

void MergeRedundant(const torch::Tensor &pos_fb,
                    const torch::Tensor &normal_rad_fb,
                    const torch::Tensor &idx_fb, torch::Tensor free_mask,
                    float max_dist, float max_angle, int neighbor_size);

torch::Tensor FindLiveToModelMerges(const torch::Tensor &live_pos_fb,
                                    const torch::Tensor &live_normal_fb,
                                    const torch::Tensor &live_idx_fb,
                                    const torch::Tensor &live_feats,
                                    const torch::Tensor &model_pos_fb,
                                    const torch::Tensor &model_normal_fb,
                                    const torch::Tensor &model_idx_fb,
                                    const torch::Tensor &model_feats,
                                    float max_normal_angle, bool use_feats);
}  // namespace fiontb
