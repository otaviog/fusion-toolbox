#pragma once

#include <unordered_map>

#include <torch/torch.h>

#include "aabb.hpp"
#include "eigen_common.hpp"

namespace fiontb {

template <typename CellType>
class HashVolume {
 public:
  typedef std::unordered_map<int64_t, CellType> MapType;
  typedef typename MapType::iterator iterator;
  typedef typename MapType::const_iterator const_iterator;

  HashVolume(const AABB &aabb, float voxel_size)
      : min_point(aabb.get_min()), voxel_size(voxel_size) {
    const Eigen::Vector3f dim_dists = aabb.get_max() - aabb.get_min();
    dims[0] = dim_dists[2] / voxel_size;
    dims[1] = dim_dists[1] / voxel_size;
    dims[2] = dim_dists[0] / voxel_size;
  }
  CellType &operator()(const Eigen::Vector3f &point) {
    return volume_[Hash(point)];
  }

  CellType &operator[](int64_t voxel_id) { return volume_[voxel_id]; }

  iterator Find(const Eigen::Vector3f &point) { return FindId(Hash(point)); }

  iterator FindId(int64_t id) { return volume_.find(id); }

  const_iterator FindId(int64_t id) const { return volume_.find(id); }

  const_iterator end() const { return volume_.end(); }

  void Contains(const Eigen::Vector3f &point) {
    return Find(point) != volume_.end();
  }

  torch::Tensor GetVoxelIDs() const {
    torch::Tensor voxel_ids =
        torch::empty({int64_t(volume_.size())},
                     torch::TensorOptions(torch::kCPU).dtype(torch::kInt32));
    auto acc = voxel_ids.accessor<int32_t, 1>();
    int count = 0;
    for (auto it = volume_.begin(); it != volume_.end(); ++it) {
      acc[count++] = it->first;
    }
    return voxel_ids;
  }

 private:
  int64_t Hash(const Eigen::Vector3f &point) {
    return Hash(point[0], point[1], point[2]);
  }
  int64_t Hash(float x, float y, float z) {
    const int col = (x - min_point[0]) / voxel_size;
    const int row = (y - min_point[1]) / voxel_size;
    const int depth = (z - min_point[2]) / voxel_size;

    const int voxel_id = depth * dims[0] * dims[1] + row * dims[2] + col;
    return voxel_id;
  }

  MapType volume_;
  Eigen::Vector3f min_point;
  float voxel_size;
  int dims[3];
};
}  // namespace fiontb
