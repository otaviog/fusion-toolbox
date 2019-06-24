#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <torch/torch.h>

#include "eigen_common.hpp"

#include "dense_volume.hpp"

namespace fiontb {

class SparseVolume {
 public:
  SparseVolume(float voxel_size, int unit_resolution = 32);

  std::shared_ptr<DenseVolume> GetUnit(const Eigen::Vector3f &xyz);

 private:
  int Hash(const Eigen::Vector3i &xyz);

  std::unordered_map<int, std::shared_ptr<DenseVolume>> units_;
  float voxel_size_;
  int unit_resolution_;
};
}  // namespace fiontb
