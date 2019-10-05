#pragma once

#include <torch/torch.h>

#include "error.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {
struct IndexMap {
  torch::Tensor position_confidence, normal_radius, color, indexmap;

  static void RegisterPybind(pybind11::module &m);

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
  torch::Tensor positions, confidences, normals, radii, colors, times;

  static void RegisterPybind(pybind11::module &m);

  void CheckDevice(const torch::Device &dev) const {
    FTB_CHECK_DEVICE(dev, positions);
    FTB_CHECK_DEVICE(dev, confidences);
    FTB_CHECK_DEVICE(dev, normals);
    FTB_CHECK_DEVICE(dev, radii);
    FTB_CHECK_DEVICE(dev, colors);
    FTB_CHECK_DEVICE(dev, times);
  }
};

struct SurfelFusionOp {
  static void MergeLive(const IndexMap &target_indexmap,
                        const IndexMap &live_indexmap, MappedSurfelModel model,
                        const torch::Tensor &rt_cam, int search_size,
                        float max_normal_angle, torch::Tensor new_surfels_map);

  static void CarveSpace(const IndexMap &model_indexmap,
                         torch::Tensor free_mask, int curr_time,
                         float stable_conf_thresh, int search_size,
                         float min_z_diff);

  static void Merge(const IndexMap &model_indexmap, torch::Tensor merge_map,
                    MappedSurfelModel model, float max_dist, float max_angle,
                    int neighbor_size, float stable_conf_thresh);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
