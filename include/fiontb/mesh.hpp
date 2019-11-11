#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace fiontb {

Eigen::Vector3f GetClosestPointTriangle(const Eigen::Vector3f &query,
                                        const Eigen::Vector3f &p0,
                                        const Eigen::Vector3f &p1,
                                        const Eigen::Vector3f &p2);

std::pair<Eigen::Vector3f, long> QueryClosestPoint(
    const Eigen::Vector3f &qpoint, const torch::Tensor &verts,
    const torch::Tensor &faces);

}  // namespace fiontb
