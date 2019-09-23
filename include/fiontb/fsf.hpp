#pragma once

#include <string>

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

struct FSFOp {
  static void MergeLive(const IndexMap &target_indexmap_params,
                        const IndexMap &live_indexmap_params,
                        const MappedSurfelModel &model_params, int search_size,
                        float max_normal_angle, torch::Tensor new_surfels_map);
};
}  // namespace fiontb
