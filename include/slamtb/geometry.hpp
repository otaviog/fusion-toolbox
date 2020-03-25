#pragma once

#include <torch/torch.h>

#include "eigen_common.hpp"

namespace slamtb {

/**
 * Find the closest point between on a mesh to a point.
 *
 * Code mostly taken from Szauer, Gabor. Game Physics Cookbook. Packt
 * Publishing Ltd, 2017. 
 *
 * @param qpoint The source point.
 * @param verts The mesh's vertices. Float32 [Nx32] tensor.
 * @param faces The mesh's faces. Int64 [Mx3] tensor.

 * @return The first return is the projected point, and the second is
 * its face index.
 */
std::pair<Eigen::Vector3f, long> GetClosestPoint(
    const Eigen::Vector3f &qpoint, const torch::Tensor &verts,
    const torch::Tensor &faces);

}  // namespace slamtb
