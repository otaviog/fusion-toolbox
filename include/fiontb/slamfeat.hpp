#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
struct SlamFeatOp {
  static int ExtractPatch(
      const torch::Tensor &anc_points_, const torch::Tensor &anc_colors_,
      const torch::Tensor &anc_mask_, const torch::Tensor &anc_cam_to_world_,
      const torch::Tensor &pos_points_, const torch::Tensor &pos_colors_,
      const torch::Tensor &pos_mask_, const torch::Tensor &pos_world_to_cam_,
      const torch::Tensor &pos_cam_to_world_, const torch::Tensor &pos_kcam_,
      const torch::Tensor &neg_colors_, float point_dist_thresh,
      torch::Tensor anc_patch_, torch::Tensor anc_d_patch_,
      torch::Tensor pos_patch_, torch::Tensor pos_d_patch_, torch::Tensor mask_,
      torch::Tensor neg_patch_, torch::Tensor neg_d_patch_,
      torch::Tensor hard_negative_);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
