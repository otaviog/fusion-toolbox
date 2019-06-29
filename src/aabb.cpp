#include "aabb.hpp"

#include "eigen_common.hpp"

#include "sat.hpp"

namespace fiontb {
AABB::AABB(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1) {
  min_[0] = std::min(p0[0], p1[0]);
  min_[1] = std::min(p0[1], p1[1]);
  min_[2] = std::min(p0[2], p1[2]);

  max_[0] = std::max(p0[0], p1[0]);
  max_[1] = std::max(p0[1], p1[1]);
  max_[2] = std::max(p0[2], p1[2]);
}

AABB::AABB(const torch::Tensor &points) {
  const auto mi = std::get<0>(points.min(0));
  const auto ma = std::get<0>(points.max(0));
  min_ = from_tensorv3f(mi.cpu());
  max_ = from_tensorv3f(ma.cpu());
}

AABB::AABB(const torch::Tensor &indices, const torch::Tensor &points) {}

bool AABB::IsInside(const Eigen::Vector3f &point) const {
  if (point[0] < min_[0] || point[1] < min_[1] || point[2] < min_[2])
    return false;

  if (point[0] > max_[0] || point[1] > max_[1] || point[2] > max_[2])
    return false;

  return true;
}

bool AABB::IsInside(const Eigen::Vector3f &point, float radius) const {
  Eigen::Vector3f closest(GetClosestPoint(point));

  const Eigen::Vector3f v = point - closest;
  return radius * radius > v.squaredNorm();
}

Eigen::Vector3f AABB::GetClosestPoint(const Eigen::Vector3f &point) const {
  Eigen::Vector3f result = point;
  result[0] = (result[0] < min_[0]) ? min_[0] : result[0];
  result[1] = (result[1] < min_[1]) ? min_[1] : result[1];
  result[2] = (result[2] < min_[2]) ? min_[2] : result[2];

  result[0] = (result[0] > max_[0]) ? max_[0] : result[0];
  result[1] = (result[1] > max_[1]) ? max_[1] : result[1];
  result[2] = (result[2] > max_[2]) ? max_[2] : result[2];
  return result;
}

bool AABB::Intersects(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                      const Eigen::Vector3f &p2) const {
  const Eigen::Vector3f v0 = p1 - p0;
  const Eigen::Vector3f v1 = p2 - p1;
  const Eigen::Vector3f v2 = p0 - p2;

  const Eigen::Vector3f bn0(1.0f, 0.0f, 0.0f);
  const Eigen::Vector3f bn1(0.0f, 1.0f, 0.0f);
  const Eigen::Vector3f bn2(0.0f, 0.0f, 1.0f);

  const Eigen::Vector3f sep_axis[] = {
      bn0,           bn1,           bn2,           v0.cross(v1),  bn0.cross(v0),
      bn0.cross(v1), bn0.cross(v2), bn1.cross(v0), bn1.cross(v1), bn1.cross(v2),
      bn2.cross(v0), bn2.cross(v1), bn2.cross(v2)};

  for (int i = 0; i < 13; ++i) {
    const Eigen::Vector3f axis = sep_axis[i];
    SATInterval aabb_interval = GetAABBInterval(*this, axis);
    SATInterval trig_interval = GetTriangleInterval(p0, p1, p2, axis);

    if (!aabb_interval.HasOverlap(trig_interval)) {
      return false;
    }
  }

  return true;
}

void SubdivideAABBOcto(const AABB &aabb, AABB subs[8]) {
  using namespace Eigen;

  const Vector3f mi = aabb.get_min();
  const Vector3f ma = aabb.get_max();
  const Vector3f ce = (mi + ma) * 0.5;

  subs[0] = AABB(Vector3f(mi[0], mi[1], mi[2]), ce);
  subs[1] = AABB(Vector3f(mi[0], mi[1], ma[2]), ce);
  subs[2] = AABB(Vector3f(mi[0], ma[1], ma[2]), ce);
  subs[3] = AABB(Vector3f(mi[0], ma[1], mi[2]), ce);
  subs[4] = AABB(Vector3f(ma[0], mi[1], mi[2]), ce);
  subs[5] = AABB(Vector3f(ma[0], mi[1], ma[2]), ce);
  subs[6] = AABB(Vector3f(ma[0], ma[1], ma[2]), ce);
  subs[7] = AABB(Vector3f(ma[0], ma[1], mi[2]), ce);
}

}  // namespace fiontb
