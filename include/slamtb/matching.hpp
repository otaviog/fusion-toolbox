#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace slamtb {

struct CorrespondenceMap {
  static void ComputeCorrespondenceMap(
      const torch::Tensor &source_points, const torch::Tensor &source_normals,
      const torch::Tensor &source_mask, const torch::Tensor &rt_cam,
      const torch::Tensor &target_points, const torch::Tensor &target_normals,
      const torch::Tensor &target_mask, const torch::Tensor &kcam,
      torch::Tensor merge_map, double distance_thresh,
      double normal_angle_thesh);

  static void RegisterPybind(pybind11::module &m);
};
struct FPCLMatcherOp {
  static void Forward(const torch::Tensor &target_points,
                      const torch::Tensor &target_normals,
                      const torch::Tensor &target_mask,
                      const torch::Tensor &target_features,
                      const torch::Tensor &source_points,
                      const torch::Tensor &source_normals,
                      const torch::Tensor &kcam, float distance_thresh,
                      float normals_angle_thresh, torch::Tensor out_points,
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
