#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
struct ICPJacobian {
  static int EstimateGeometric(
      const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
      const torch::Tensor &tgt_mask, const torch::Tensor &src_points,
      const torch::Tensor &src_normals, const torch::Tensor &src_mask,
      const torch::Tensor &kcam, const torch::Tensor &rt_cam,
      torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
      torch::Tensor squared_residual);

  static int EstimateFeature(
      const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
      const torch::Tensor &tgt_feat, const torch::Tensor &tgt_mask,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &kcam,
      const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
      torch::Tensor Jtr_partial, torch::Tensor squared_residual);

  static int EstimateFeatureSO3(
      const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
      const torch::Tensor &tgt_feat, const torch::Tensor &tgt_mask,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &kcam,
      const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
      torch::Tensor Jtr_partial, torch::Tensor squared_residual);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
