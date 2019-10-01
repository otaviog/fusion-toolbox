#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace fiontb {
struct ICPJacobian {
  static void EstimateGeometric(const torch::Tensor points0,
                                const torch::Tensor normals0,
                                const torch::Tensor mask0,
                                const torch::Tensor points1,
                                const torch::Tensor mask1,
                                const torch::Tensor kcam,
                                const torch::Tensor params,
                                torch::Tensor jacobian, torch::Tensor residual);

  static void EstimateHybrid(
      const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
      const torch::Tensor tgt_feat, const torch::Tensor tgt_mask,
      const torch::Tensor src_points, const torch::Tensor src_feats,
      const torch::Tensor src_mask, const torch::Tensor kcam,
      const torch::Tensor rt_cam, float geom_weight, float feat_weight,
      torch::Tensor jacobian, torch::Tensor residual);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
