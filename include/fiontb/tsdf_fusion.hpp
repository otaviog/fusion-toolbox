#pragma once

#include <memory>

#include "eigen_common.hpp"

#include "dense_volume.hpp"
#include "sparse_volume.hpp"

namespace fiontb {
void FuseDenseVolume(std::shared_ptr<DenseVolume> volume,
                     const torch::Tensor &wpos_image,
                     const Eigen::Vector3f &cam_center,
                     const Eigen::Matrix4f &world_to_cam,
                     const Eigen::Matrix3f &cam_project, float tsdf_trunc);

void FuseSparseVolume(std::shared_ptr<SparseVolume> volume,
                      const torch::Tensor &wpoints_image,
                      const Eigen::Vector3f &cam_center,
                      const Eigen::Matrix4f &world_to_cam,
                      const Eigen::Matrix3f &cam_project, float tsdf_trunc);

}  // namespace fiontb
