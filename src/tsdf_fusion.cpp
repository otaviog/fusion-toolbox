#include "tsdf_fusion.hpp"

#include <set>

using namespace std;

namespace fiontb {

void FuseDenseVolume(shared_ptr<DenseVolume> volume,
                     const torch::Tensor &wpos_image,
                     const Eigen::Vector3f &cam_center,
                     const Eigen::Matrix4f &world_to_cam,
                     const Eigen::Matrix3f &cam_project, float tsdf_trunc) {
  const auto wpos_acc = wpos_image.accessor<float, 3>();
  auto sdf_acc = volume->sdf.accessor<float, 3>();
  auto weight_acc = volume->weights.accessor<float, 3>();

  for (int i = 0; i < volume->resolution; ++i) {
    for (int j = 0; j < volume->resolution; ++j) {
      for (int k = 0; k < volume->resolution; ++k) {
        const Eigen::Vector3f _world_pos = volume->get_world_pos(i, j, k);
        const Eigen::Vector4f world_pos(_world_pos[0], _world_pos[1],
                                        _world_pos[2], 1.0);

        const Eigen::Vector4f _camera_pos = world_to_cam * world_pos;
        if (_camera_pos[3] <= 0) continue;

        const Eigen::Vector3f camera_pos(_camera_pos[0], _camera_pos[1],
                                         _camera_pos[2]);

        Eigen::Vector3f frame_pos = cam_project * camera_pos;
        frame_pos /= camera_pos[2];

        const int col = frame_pos[0];
        if (col < 0 || col >= wpos_image.size(1)) continue;

        const int row = frame_pos[1];
        if (row < 0 || row >= wpos_image.size(0)) continue;

        const Eigen::Vector3f view_vec(_world_pos - cam_center);

        const float sdf = wpos_acc[row][col][2] - view_vec.norm();

        if (sdf >= -tsdf_trunc) {
          const float tsdf = min(1.0f, sdf / tsdf_trunc);
          const float w = 1.0f;

          const float curr_tsdf = sdf_acc[i][j][k];
          const float curr_weight = weight_acc[i][j][k];
          sdf_acc[i][j][k] =
              (curr_tsdf * curr_weight + w * tsdf) / (curr_weight + w);
        }
      }
    }
  }
}

void FuseSparseVolume(shared_ptr<SparseVolume> volume,
                      const torch::Tensor &wpoints_image,
                      const Eigen::Vector3f &cam_center,
                      const Eigen::Matrix4f &world_to_cam,
                      const Eigen::Matrix3f &cam_project, float tsdf_trunc) {
  const torch::Tensor world_points = wpoints_image.view({-1, 3});
  auto wp_acc = world_points.accessor<float, 2>();
  set<DenseVolume *> visited_set;

  for (long row = 0; row < world_points.size(0); ++row) {
    const Eigen::Vector3f wpoint(wp_acc[row][0], wp_acc[row][1],
                                 wp_acc[row][2]);
    auto vol_unit = volume->GetUnit(wpoint);

    if (visited_set.count(vol_unit.get())) continue;

    visited_set.insert(vol_unit.get());
    FuseDenseVolume(vol_unit, wpoints_image, cam_center, world_to_cam,
                    cam_project, tsdf_trunc);
  }
}

}  // namespace fiontb
