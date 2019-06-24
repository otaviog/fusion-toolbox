#include "sparse_volume.hpp"

using namespace std;

namespace fiontb {
SparseVolume::SparseVolume(float voxel_size, int unit_resolution) {
  voxel_size_ = voxel_size;
  unit_resolution_ = unit_resolution;
}

int SparseVolume::Hash(const Eigen::Vector3i &ijk) {
  return ijk[0] * 512 * 512 + ijk[1] * 512 + ijk[0];
}

shared_ptr<DenseVolume> SparseVolume::GetUnit(const Eigen::Vector3f &_xyz) {
  const Eigen::Vector3i unit_ijk(
      floor(_xyz[0] / voxel_size_ + 0.5) / unit_resolution_,
      floor(_xyz[1] / voxel_size_ + 0.5) / unit_resolution_,
      floor(_xyz[2] / voxel_size_ + 0.5) / unit_resolution_);

  const int key = Hash(unit_ijk);
  auto found = units_.find(key);
  if (found != units_.end()) {
    return found->second;
  }

  auto unit = make_shared<DenseVolume>(unit_resolution_, voxel_size_, unit_ijk);
  units_[key] = unit;

  return unit;
}

}  // namespace fiontb
