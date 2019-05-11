#include "aabb.hpp"

#include "eigen_common.hpp"

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

void GPUCreateAABBFromPoints(const torch::Tensor &indices, const torch::Tensor &points,
                             float min[3], float max[3]);

AABB::AABB(const torch::Tensor &indices, const torch::Tensor &points) {
  
}


bool AABB::IsInside(const Eigen::Vector3f &point) const {
  return (min_[0] >= point[0] && max_[0] <= point[0]) &&
         (min_[1] >= point[1] && max_[1] <= point[1]) &&
         (min_[2] >= point[2] && max_[2] <= point[2]);
}

bool AABB::IsInside(const Eigen::Vector3f &point, float radius) const {
  Eigen::Vector3f closest(GetClosestPoint(point));

  const Eigen::Vector3f v = point - closest;
  return v.squaredNorm() <= radius*radius;
}

torch::Tensor GPUIsInside(const torch::Tensor &indices,
                          const torch::Tensor &points, float min[3],
                          float max[3]);

torch::Tensor AABB::IsInside(const torch::Tensor &indices,
                             const torch::Tensor &points) const {
  float min[3] = {min_[0], min_[1], min_[2]};
  float max[3] = {max_[0], max_[1], max_[2]};

  return GPUIsInside(indices, points, min, max);
}

torch::Tensor GPUIsInside(const torch::Tensor &points, float min[3],
                           float max[3]);

torch::Tensor AABB::IsInside(const torch::Tensor &points) const {
  float min[3] = {min_[0], min_[1], min_[2]};
  float max[3] = {max_[0], max_[1], max_[2]};

  return GPUIsInside(points, min, max);
}

torch::Tensor GPUIsInside(const torch::Tensor &points, float radius,
                           float min[3], float max[3]);

torch::Tensor AABB::IsInside(const torch::Tensor &points, float radius) const {
  float min[3] = {min_[0], min_[1], min_[2]};
  float max[3] = {max_[0], max_[1], max_[2]};

  return GPUIsInside(points, radius, min, max);
}

torch::Tensor GPUIsInside(const torch::Tensor &indices,
                           const torch::Tensor &points, float radius,
                           float min[3], float max[3]);

torch::Tensor AABB::IsInside(const torch::Tensor &indices,
                             const torch::Tensor &points, float radius) const {
  float min[3] = {min_[0], min_[1], min_[2]};
  float max[3] = {max_[0], max_[1], max_[2]};

  return GPUIsInside(indices, points, radius, min, max);
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

torch::Tensor AABB::GetClosestPoint(const torch::Tensor &points) {
  auto tmin = torch::from_blob(min_.data(), {3}).to(points.device());
  auto tmax = torch::from_blob(max_.data(), {3}).to(points.device());

  return points.min(tmax).max(tmin);
}

}  // namespace fiontb
