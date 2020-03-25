#pragma once

#include <unordered_map>

#include <torch/torch.h>

#include "aabb.hpp"
#include "eigen_common.hpp"

namespace slamtb {

template <typename CellType>
class HashVolume {
 public:
  typedef std::unordered_map<int64_t, CellType> MapType;
  typedef typename MapType::iterator iterator;
  typedef typename MapType::const_iterator const_iterator;

  HashVolume(const Eigen::Vector3f &min_point, const Eigen::Vector3f &max_point,
             float voxel_size)
      : min_point_(min_point), voxel_size_(voxel_size) {
    const Eigen::Vector3f dim_dists = max_point - min_point;
    depth = dim_dists[2] / voxel_size_;
    height = dim_dists[1] / voxel_size_;
    width = dim_dists[0] / voxel_size_;
  }

  HashVolume(const AABB &aabb, float voxel_size)
      : min_point_(aabb.get_min()), voxel_size_(voxel_size) {
    const Eigen::Vector3f dim_dists = aabb.get_max() - aabb.get_min();
    depth = dim_dists[2] / voxel_size_;
    height = dim_dists[1] / voxel_size_;
    width = dim_dists[0] / voxel_size_;
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
    const int col = (x - min_point_[0]) / voxel_size_;
    const int row = (y - min_point_[1]) / voxel_size_;
    const int depth = (z - min_point_[2]) / voxel_size_;

    const int voxel_id = depth * width * height + row * width + col;
    return voxel_id;
  }

  MapType volume_;
  Eigen::Vector3f min_point_;
  float voxel_size_;
  int depth, height, width;
};
}  // namespace slamtb
