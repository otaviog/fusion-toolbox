#pragma once

#include <torch/torch.h>

#include "error.hpp"

namespace fiontb {

void CarveSpace(const torch::Tensor model_pos_fb,
                const torch::Tensor model_idx_fb, torch::Tensor mask,
                int curr_time, float stable_conf_thresh, int search_size,
                float min_z_diff);

void FindMergeableSurfels(const torch::Tensor &pos_fb,
                          const torch::Tensor &normal_rad_fb,
                          const torch::Tensor &idx_fb, torch::Tensor merge_map,
                          float max_dist, float max_angle, int neighbor_size,
                          float stable_conf_thresh);

torch::Tensor FindLiveToModelMerges(const torch::Tensor &live_pos_fb,
                                    const torch::Tensor &live_normal_fb,
                                    const torch::Tensor &live_idx_fb,
                                    const torch::Tensor &model_pos_fb,
                                    const torch::Tensor &model_normal_fb,
                                    const torch::Tensor &model_idx_fb,
                                    float max_normal_angle, int search_size);

torch::Tensor FindFeatLiveToModelMerges(
    const torch::Tensor &live_pos_fb, const torch::Tensor &live_normal_fb,
    const torch::Tensor &live_idx_fb, const torch::Tensor &live_feats,
    const torch::Tensor &model_pos_fb, const torch::Tensor &model_normal_fb,
    const torch::Tensor &model_idx_fb, const torch::Tensor &model_feats,
    float max_normal_angle, int search_size);

inline void CheckDevice(const torch::Device expected_dev,
                        const torch::Tensor &test_tensor, const char *file,
                        int line) {
  if (expected_dev != test_tensor.device()) {
    if (test_tensor.is_cuda()) {
      Check(false, file, line, "Expected a cpu tensor");
    } else {
      Check(false, file, line, "Expected a gpu tensor");
    }
  }
}

#define FTB_CHECK_DEVICE(device, test_tensor) \
  fiontb::CheckDevice(device, test_tensor, __FILE__, __LINE__)

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

};  // namespace fiontb

struct FeatSurfel {
  static void MergeLive(const IndexMap &target_indexmap_params,
                        const IndexMap &live_indexmap_params,
                        const MappedSurfelModel &model_params, int search_size,
                        float max_normal_angle, torch::Tensor new_surfels_map);
};

}  // namespace fiontb
