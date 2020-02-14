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
  static int GenerateTriplet(const SFFrame &anch_pcl,
                             const torch::Tensor anch_cam_to_world,
                             const SFFrame &posv_pcl,
                             const torch::Tensor &posv_kcam,
                             const torch::Tensor posv_cam_to_world,
                             float point_dist_thresh, const SFFrame &negv_pcl,
                             SFCrop anch_crop, SFCrop posv_crop,
                             torch::Tensor posv_mask, SFCrop negv_crop,
                             torch::Tensor negv_mask, torch::Tensor negv_hard);

  static int GenerateTriplet2(
      const SFFrame &anch_frame, const torch::Tensor anch_cam_to_world,
      const SFFrame &posv_frame, const torch::Tensor &posv_kcam,
      const torch::Tensor posv_cam_to_world, float point_dist_thresh,
      const SFFrame &negv_frame, SFCrop anch_crop, SFCrop posv_crop,
      torch::Tensor posv_indices, torch::Tensor posv_mask, SFCrop negv_crop,
      torch::Tensor negv_mask, torch::Tensor negv_hard);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
