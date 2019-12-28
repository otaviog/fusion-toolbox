#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {

struct SFFrame {
  torch::Tensor mask, points, colors, depths;
};

struct SFCrop {
  torch::Tensor colors, depths;
};

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

  static int GenerateTriplet(const SFFrame &anch_pcl,
                             const torch::Tensor anch_cam_to_world,
                             const SFFrame &posv_pcl,
                             const torch::Tensor &posv_kcam,
                             const torch::Tensor posv_cam_to_world,
                             float point_dist_thresh, const SFFrame &negv_pcl,
                             SFCrop anch_crop, SFCrop posv_crop,
                             torch::Tensor posv_mask, SFCrop negv_crop,
                             torch::Tensor negv_mask, torch::Tensor negv_hard);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
