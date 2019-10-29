#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
struct ICPJacobian {
  static int EstimateGeometric(
      const torch::Tensor &points0, const torch::Tensor &normals0,
      const torch::Tensor &mask0, const torch::Tensor &points1,
      const torch::Tensor &mask1, const torch::Tensor &kcam,
      const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
      torch::Tensor Jr_partial, torch::Tensor squared_residual);

  static int EstimateFeature(
      const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
      const torch::Tensor &tgt_feat, const torch::Tensor &tgt_mask,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &kcam,
      const torch::Tensor &rt_cam, torch::Tensor JtJ_partial, torch::Tensor Jtr_partial,
      torch::Tensor squared_residual);

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
