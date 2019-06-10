#pragma once

#include "eigen_common.hpp"

#include "aabb.hpp"

namespace fiontb {
struct SATInterval {
  SATInterval(float min = 1.0f, float max = -1.0f) : min(min), max(max) {}

  bool HasOverlap(const SATInterval &other) const;
  
  float min, max;
};

SATInterval GetTriangleInterval(const Eigen::Vector3f &p0,
                                const Eigen::Vector3f &p1,
                                const Eigen::Vector3f &p2,
                                const Eigen::Vector3f &axis);

SATInterval GetAABBInterval(const AABB &aabb, const Eigen::Vector3f &axis);

}  // namespace fiontb
