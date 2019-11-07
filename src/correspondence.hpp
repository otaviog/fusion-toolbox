#pragma once

#include "accessor.hpp"
#include "camera.hpp"

namespace fiontb {
template <Device dev, typename scalar_t>
struct SimpleCorrespondence {
  const PointGrid<dev, scalar_t> tgt;
  const KCamera<dev, scalar_t> kcam;

  SimpleCorrespondence(const torch::Tensor &points, const torch::Tensor &normals,
                      const torch::Tensor &mask, const torch::Tensor kcam)
      : tgt(points, normals, mask), kcam(kcam) {}
  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal) const {
    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    Eigen::Vector2i src_uv = kcam.Project(src_point);
    if (src_uv[0] < 0 || src_uv[0] >= width || src_uv[1] < 0 ||
        src_uv[1] >= height)
      return false;

    if (tgt.empty(src_uv[1], src_uv[0])) return false;

    tgt_point = to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]);
    tgt_normal = to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]);
    return true;
  }
};

template <Device dev, typename scalar_t>
struct RobustCorrespondence {
  const PointGrid<dev, scalar_t> tgt;
  const KCamera<dev, scalar_t> kcam;
  const float distance_thresh;
  const float angle_thresh;

  RobustCorrespondence(const torch::Tensor &points, const torch::Tensor &normals,
                      const torch::Tensor &mask, const torch::Tensor kcam,
                      float distance_thresh = .1f,
                      float angle_thresh = sin(20.f * 3.14159254f / 180.f))
      : tgt(points, normals, mask),
        kcam(kcam),
        distance_thresh(distance_thresh * distance_thresh),
        angle_thresh(angle_thresh) {}

  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             const Vector<scalar_t, 3> &src_normal,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal) const {
    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    Eigen::Vector2i src_uv = kcam.Project(src_point);
    if (src_uv[0] < 0 || src_uv[0] >= width || src_uv[1] < 0 ||
        src_uv[1] >= height)
      return false;

    if (tgt.empty(src_uv[1], src_uv[0])) return false;

    tgt_point = to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]);
    if ((tgt_point - src_point).squaredNorm() > distance_thresh) return false;

    tgt_normal = to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]);
    const scalar_t sine = src_normal.cross(tgt_normal).norm();
    if (sine >= angle_thresh) return false;

    return true;
  }
};
}  // namespace fiontb
