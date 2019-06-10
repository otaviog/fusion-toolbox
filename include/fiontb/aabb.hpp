#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace fiontb {
class AABB {
 public:
  AABB(const Eigen::Vector3f &p0 = Eigen::Vector3f(0.0, 0.0, 0.0),
       const Eigen::Vector3f &p1 = Eigen::Vector3f(0.0, 0.0, 0.0));
  AABB(const torch::Tensor &points);
  AABB(const torch::Tensor &indices, const torch::Tensor &points);

  Eigen::Vector3f get_min() const { return min_; }

  Eigen::Vector3f get_max() const { return max_; }

  bool IsInside(const Eigen::Vector3f &point) const;

  bool IsInside(const Eigen::Vector3f &point, float radius) const;

  torch::Tensor IsInside(const torch::Tensor &indices,
                         const torch::Tensor &points) const;

  torch::Tensor IsInside(const torch::Tensor &points) const;

  torch::Tensor IsInside(const torch::Tensor &points, float radius) const;

  torch::Tensor IsInside(const torch::Tensor &indices,
                         const torch::Tensor &points, float radius) const;

  Eigen::Vector3f GetClosestPoint(const Eigen::Vector3f &point) const;

  torch::Tensor GetClosestPoint(const torch::Tensor &points);

  bool Intersects(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                  const Eigen::Vector3f &p2) const;

 private:
  Eigen::Vector3f min_, max_;
};

void SubdivideAABBOcto(const AABB &aabb, AABB subs[8]);

}  // namespace fiontb
