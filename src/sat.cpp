#include "sat.hpp"

using namespace std;

namespace fiontb {
SATInterval GetTriangleInterval(const Eigen::Vector3f &p0,
                                const Eigen::Vector3f &p1,
                                const Eigen::Vector3f &p2,
                                const Eigen::Vector3f &axis) {
  SATInterval interval;
  interval.min = axis.dot(p0);
  interval.max = interval.min;

  const float dot1 = axis.dot(p1);
  interval.min = min(interval.min, dot1);
  interval.max = max(interval.max, dot1);

  const float dot2 = axis.dot(p2);
  interval.min = min(interval.min, dot2);
  interval.max = max(interval.max, dot2);

  return interval;
}

SATInterval GetAABBInterval(const AABB &aabb, const Eigen::Vector3f &axis) {
  const Eigen::Vector3f bmin = aabb.get_min();
  const Eigen::Vector3f bmax = aabb.get_max();
  SATInterval interval;

  Eigen::Vector3f corners[] = {
      Eigen::Vector3f(bmin[0], bmin[1], bmin[2]),
      Eigen::Vector3f(bmax[0], bmin[1], bmin[2]),
      Eigen::Vector3f(bmin[0], bmax[1], bmin[2]),
      Eigen::Vector3f(bmax[0], bmax[1], bmin[2]),
      Eigen::Vector3f(bmin[0], bmin[1], bmax[2]),
      Eigen::Vector3f(bmax[0], bmin[1], bmax[2]),
      Eigen::Vector3f(bmin[0], bmax[1], bmax[2]),
      Eigen::Vector3f(bmax[0], bmax[1], bmax[2]),
  };
  interval.min = axis.dot(corners[0]);
  interval.max = interval.min;
  for (int i = 1; i < 8; ++i) {
    const Eigen::Vector3f &corner = corners[i];
    const float dotv = axis.dot(corner);

    interval.min = min(interval.min, dotv);
    interval.max = max(interval.max, dotv);
  }

  return interval;
}

bool SATInterval::HasOverlap(const SATInterval &other) const {
  return (other.min <= this->max) && (this->min <= other.max);
}
}  // namespace fiontb
