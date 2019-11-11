#pragma once

#include "accessor.hpp"
#include "camera.hpp"
#include "pointgrid.hpp"

namespace fiontb {
template <Device dev, typename scalar_t>
class PointGrid : public BasePointGrid<dev> {
 public:
  const typename Accessor<dev, scalar_t, 3>::T points;
  const typename Accessor<dev, scalar_t, 3>::T normals;

  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
            const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, scalar_t, 3>::Get(points)),
        normals(Accessor<dev, scalar_t, 3>::Get(normals)) {}
};

template <Device dev, typename scalar_t>
struct SimpleCorrespondence {
  const PointGrid<dev, scalar_t> tgt;
  const KCamera<dev, scalar_t> kcam;

  SimpleCorrespondence(const torch::Tensor &points,
                       const torch::Tensor &normals, const torch::Tensor &mask,
                       const torch::Tensor &kcam)
      : tgt(points, normals, mask), kcam(kcam) {}
  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal) const {
    Eigen::Vector2i src_uv = kcam.Project(src_point);
    if (src_uv[0] < 0 || src_uv[0] >= tgt.width || src_uv[1] < 0 ||
        src_uv[1] >= tgt.height)
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
  const scalar_t distance_thresh;
  const scalar_t angle_thresh;

  RobustCorrespondence(const torch::Tensor &points,
                       const torch::Tensor &normals, const torch::Tensor &mask,
                       const torch::Tensor &kcam, double distance_thresh = 0.1,
                       double angle_thresh = sin(20. * 3.14159254 / 180.))
      : tgt(points, normals, mask),
        kcam(kcam),
        distance_thresh(distance_thresh * distance_thresh),
        angle_thresh(angle_thresh) {}

  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             const Vector<scalar_t, 3> &src_normal,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal) const {
    Eigen::Vector2i src_uv = kcam.Project(src_point);
    if (src_uv[0] < 0 || src_uv[0] >= tgt.width || src_uv[1] < 0 ||
        src_uv[1] >= tgt.height)
      return false;

    if (tgt.empty(src_uv[1], src_uv[0])) return false;

    tgt_point = to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]);
    if ((tgt_point - src_point).squaredNorm() > distance_thresh) return false;

    tgt_normal = to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]);
    const scalar_t sine = src_normal.cross(tgt_normal).norm();
    if (sine >= angle_thresh) return false;

    return true;
  }

  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal) const {
    Eigen::Vector2i src_uv = kcam.Project(src_point);
    if (src_uv[0] < 0 || src_uv[0] >= tgt.width || src_uv[1] < 0 ||
        src_uv[1] >= tgt.height)
      return false;

    if (tgt.empty(src_uv[1], src_uv[0])) return false;

    tgt_point = to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]);
    if ((tgt_point - src_point).squaredNorm() > distance_thresh) return false;

    tgt_normal = to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]);
    return true;
  }

  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point,
                             Vector<scalar_t, 3> &tgt_point,
                             Vector<scalar_t, 3> &tgt_normal, scalar_t &u,
                             scalar_t &v) const {
    kcam.Project(src_point, u, v);
    const int ui = int(round(u));
    const int vi = int(round(v));

    if (ui < 0 || ui >= tgt.width || vi < 0 || vi >= tgt.height) return false;

    if (tgt.empty(vi, ui)) return false;

    tgt_point = to_vec3<scalar_t>(tgt.points[vi][ui]);
    if ((tgt_point - src_point).squaredNorm() > distance_thresh) return false;

    tgt_normal = to_vec3<scalar_t>(tgt.normals[vi][ui]);
    return true;
  }

  FTB_DEVICE_HOST bool Match(const Vector<scalar_t, 3> &src_point, scalar_t &u,
                             scalar_t &v) const {
    kcam.Project(src_point, u, v);
    const int ui = int(round(u));
    const int vi = int(round(v));

    if (ui < 0 || ui >= tgt.width || vi < 0 || vi >= tgt.height) return false;
    if (tgt.empty(vi, ui)) return false;

    const Vector<scalar_t, 3> tgt_point = to_vec3<scalar_t>(tgt.points[vi][ui]);
    // if ((tgt_point - src_point).squaredNorm() > distance_thresh) return
    // false;

    return true;
  }
};
}  // namespace fiontb
