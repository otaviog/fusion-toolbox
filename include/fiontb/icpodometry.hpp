#pragma once

#include <vector>

#include <torch/torch.h>

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

  static void EstimateIntensity(
      const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
      const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
      const torch::Tensor tgt_mask, const torch::Tensor src_points,
      const torch::Tensor src_intensity, const torch::Tensor src_mask,
      const torch::Tensor kcam, const torch::Tensor rt_cam,
      torch::Tensor jacobian, torch::Tensor residual);
};
}  // namespace fiontb
