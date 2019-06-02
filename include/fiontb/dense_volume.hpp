#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace fiontb {
class DenseVolume {
 public:
  DenseVolume(int resolution, float voxel_size,
              Eigen::Vector3i begin_pos = Eigen::Vector3i(0, 0, 0));

  torch::Tensor ToPointCloud();
  
  Eigen::Vector3f get_world_pos(int i, int j, int k) const;

  const int resolution;
  const float voxel_size;
  const Eigen::Vector3i begin_pos;

  torch::Tensor sdf;
  torch::Tensor weights;
};

}  // namespace fiontb
