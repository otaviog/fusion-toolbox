#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace slamtb {

struct FPCLMatcherOp {
  static void Forward(const torch::Tensor &target_points,
                      const torch::Tensor &target_normals,
                      const torch::Tensor &target_mask,
                      const torch::Tensor &target_features,
                      const torch::Tensor &source_points,
                      const torch::Tensor &source_normals,
                      const torch::Tensor &kcam,
                      float distance_thresh, float normals_angle_thresh,
                      torch::Tensor out_points,
                      torch::Tensor out_normals, torch::Tensor out_features,
                      torch::Tensor match_mask);

  static void Backward(const torch::Tensor &target_features,
                       const torch::Tensor &source_points,
                       const torch::Tensor &match_mask,
                       const torch::Tensor &dl_feature,
                       const torch::Tensor &kcam, double grad_precision,
                       torch::Tensor dx_points);

  static void RegisterPybind(pybind11::module &m);
};

}  // namespace slamtb
