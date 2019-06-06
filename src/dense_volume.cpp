#include "dense_volume.hpp"

using namespace std;

namespace fiontb {
DenseVolume::DenseVolume(int resolution, float voxel_size,
                         Eigen::Vector3i begin_pos)
    : resolution(resolution), voxel_size(voxel_size), begin_pos(begin_pos) {
  sdf = torch::full({resolution, resolution, resolution}, -1.0, torch::kFloat);
  weights = torch::zeros({resolution, resolution, resolution}, torch::kFloat);
}

Eigen::Vector3f DenseVolume::get_world_pos(int i, int j, int k) const {
  return Eigen::Vector3f(
      i * voxel_size + begin_pos[0] * resolution * voxel_size,
      j * voxel_size + begin_pos[1] * resolution * voxel_size,
      k * voxel_size + begin_pos[2] * resolution * voxel_size);
}

struct XYZ {
  float x, y, z;
};

torch::Tensor DenseVolume::ToPointCloud() {
  std::vector<XYZ> pcl;
  auto sdf_acc = sdf.accessor<float, 3>();
  auto weight_acc = weights.accessor<float, 3>();

  for (int i = 0; i < resolution; ++i) {
    for (int j = 0; j < resolution; ++j) {
      for (int k = 0; k < resolution; ++k) {
        const float sdf = sdf_acc[i][j][k];
        if (weight_acc[i][j][k] != 0.0f && sdf < 0.98f && sdf >= -0.98f) {
          Eigen::Vector3f xyz(get_world_pos(i, j, k));
          XYZ xyz_store;
          xyz_store.x = xyz[0];
          xyz_store.y = xyz[1];
          xyz_store.z = xyz[2];

          pcl.push_back(xyz_store);
        }
      }
    }
  }

  return torch::from_blob(&pcl[0], {long(pcl.size()), 3});
}

}  // namespace fiontb
