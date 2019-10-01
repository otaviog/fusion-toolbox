#pragma once

#include <torch/torch.h>

#include "error.hpp"

namespace fiontb {
struct IndexMap {
  torch::Tensor position_confidence, normal_radius, color, indexmap;

  int get_width() const { return position_confidence.size(1); }

  int get_height() const { return position_confidence.size(0); }

  const torch::Device get_device() const {
    return position_confidence.device();
  }

  IndexMap To(const std::string &dev) const {
    IndexMap result;
    result.position_confidence = position_confidence.to(dev);
    result.normal_radius = normal_radius.to(dev);
    result.color = color.to(dev);
    result.indexmap = indexmap.to(dev);
    return result;
  }

  void CheckDevice(const torch::Device &dev) const {
    FTB_CHECK_DEVICE(dev, position_confidence);
    FTB_CHECK_DEVICE(dev, normal_radius);
    FTB_CHECK_DEVICE(dev, color);
    FTB_CHECK_DEVICE(dev, indexmap);
  }
};

struct MappedSurfelModel {
  torch::Tensor positions, confidences, normals, radii, colors;

  void CheckDevice(const torch::Device &dev) const {
    FTB_CHECK_DEVICE(dev, positions);
    FTB_CHECK_DEVICE(dev, confidences);
    FTB_CHECK_DEVICE(dev, normals);
    FTB_CHECK_DEVICE(dev, radii);
    FTB_CHECK_DEVICE(dev, colors);
  }
};

struct SurfelCloud {
  torch::Tensor positions, confidences, normals, radii, colors;
};

struct SurfelFusionOp {
  static void CarveSpace(const torch::Tensor &model_pos_fb,
                         const torch::Tensor model_idx_fb, torch::Tensor mask,
                         int curr_time, float stable_conf_thresh,
                         int search_size, float min_z_diff);

  static void FindMergeableSurfels(const torch::Tensor &pos_fb,
                                   const torch::Tensor &normal_rad_fb,
                                   const torch::Tensor &idx_fb,
                                   torch::Tensor merge_map, float max_dist,
                                   float max_angle, int neighbor_size,
                                   float stable_conf_thresh);

  static torch::Tensor FindLiveToModelMerges(
      const torch::Tensor &live_pos_fb, const torch::Tensor &live_normal_fb,
      const torch::Tensor &live_idx_fb, const torch::Tensor &model_pos_fb,
      const torch::Tensor &model_normal_fb, const torch::Tensor &model_idx_fb,
      float max_normal_angle, int search_size);
};
}  // namespace fiontb
