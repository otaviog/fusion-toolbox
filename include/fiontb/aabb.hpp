#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace fiontb {
/**
 * Simple Axis Aligned Bounding Box class.
 */
class AABB {
 public:
  /**
   * Initialization.
   *
   * @param p0 minimum point.
   * @param p0 maximum point.
   */
  AABB(const Eigen::Vector3f &p0 = Eigen::Vector3f(0.0, 0.0, 0.0),
       const Eigen::Vector3f &p1 = Eigen::Vector3f(0.0, 0.0, 0.0));
  /**
   * Find the minimum and maximum bounds from a point array.
   *
   * @param points Array of points. Float32 [Nx3] tensor.
   */
  AABB(const torch::Tensor &points);

  /**
   * Bounding box minimum point.
   */
  Eigen::Vector3f get_min() const { return min_; }

  /**
   * Bounding box maximum point.
   */
  Eigen::Vector3f get_max() const { return max_; }

  /**
   * Test whatever if a point is inside of the box.
   *
   * @param point The point to be tested.
   *
   * @return true if it is inside.
   */
  bool IsInside(const Eigen::Vector3f &point) const;

  /**
   * Test whatever if a sphere is intersecs with the box.
   *
   * @param point The sphere's center.
   * @param radius The sphere's radius.
   *
   * @return true if it does intersects.
   */
  bool Intersects(const Eigen::Vector3f &point, float radius) const;

  /**
   * Test whatever if a triangle intersects with the box.
   *
   * @param p0 Triangle point 0.
   * @param p0 Triangle point 1.
   * @param p0 Triangle point 2.
   *
   * @return true if it does intersects.
   */
  bool Intersects(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                  const Eigen::Vector3f &p2) const;

  /**
   * Project a point into the box's surface.
   *
   * @param point The given point.
   *
   * @return The projected point.
   */
  Eigen::Vector3f GetClosestPoint(const Eigen::Vector3f &point) const;

 private:
  Eigen::Vector3f min_, max_;
};




}  // namespace fiontb
