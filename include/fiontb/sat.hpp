/**
 * Separating Axis Theorem for testing if two shapes overlap. All the
 * code was based on the book from Szauer, Gabor. Game Physics
 * Cookbook. Packt Publishing Ltd, 2017.
 *
 */
#pragma once

#include "eigen_common.hpp"

#include "aabb.hpp"

namespace fiontb {
/**
 * Separating Axis Theorem two intervals. 
 *
 * Users should extract intervals from shapes and compare them using
 * this class.
 */
struct SATInterval {
  SATInterval(float min = 1.0f, float max = -1.0f) : min(min), max(max) {}

  /**
   * Tests if the interval overlaps with another one. 
   *
   * @param other The other one.
   *
   * @return Returns true if does overlap.
   */
  bool HasOverlap(const SATInterval &other) const;
  
  float min, /**< interval start*/
    max; /**< interval end.*/ 
};

/**
 * Create a SAT interval of a triangle on to an axis.
 *
 * @param p0 The triangle point 0.
 * @param p1 The triangle point 1.
 * @param p1 The triangle point 2.
 * @param axis The separating axis.
 *
 * @return the SAT interval. 
 */
SATInterval GetTriangleInterval(const Eigen::Vector3f &p0,
                                const Eigen::Vector3f &p1,
                                const Eigen::Vector3f &p2,
                                const Eigen::Vector3f &axis);

/**
 * Creates a SAT interval of a Axis Aligned Bounding Box on to an axis.
 *
 * @param aabb The box.
 * @param The separating axis.
 *
 * @return the SAT interval.
 */
SATInterval GetAABBInterval(const AABB &aabb, const Eigen::Vector3f &axis);

}  // namespace fiontb
