#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace fiontb {
class AABB {
 public:
  AABB(const Eigen::Vector3f &p0 = Eigen::Vector3f(0.0, 0.0, 0.0),
       const Eigen::Vector3f &p1 = Eigen::Vector3f(0.0, 0.0, 0.0));
  AABB(const torch::Tensor &points);

  Eigen::Vector3f get_min() const { return min_; }

  Eigen::Vector3f get_max() const { return max_; }

  bool IsInside(const Eigen::Vector3f &point) const;

  torch::Tensor IsInside(const torch::Tensor &indices,
                         const torch::Tensor &points) const;

  torch::Tensor IsInside(const torch::Tensor &points) const;

  torch::Tensor IsInside(const torch::Tensor &points, float radius) const;
  
  Eigen::Vector3f GetClosestPoint(const Eigen::Vector3f &point) const;

  torch::Tensor GetClosestPoint(const torch::Tensor &points);

 private:
  Eigen::Vector3f min_, max_;
};
}  // namespace fiontb
