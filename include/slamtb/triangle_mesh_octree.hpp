#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace slamtb {
namespace detail {
class AbstractOctNode;
}

/**
 * Octree structure for fast querying the closest point on a triangle
 * mesh surface to a given point.
 */
class TriangleMeshOctree {
 public:
  /**
   * Construct the Octree structure.
   *
   * It'll subdivide the scene until the leaf have the desired minimum
   * number of triangles.
   *
   * @param verts The mesh vertices. Float32 [Nx3] tensor.
   * @param faces The mesh faces. Int64 [Nx3] tensor.
   * @param leaf_num_trigs Leaf node number of triangles. Subdivision
   * will stop when this number is reach.
   */
  TriangleMeshOctree(torch::Tensor verts, const torch::Tensor &faces,
                     int leaf_num_trigs);
  ~TriangleMeshOctree();

  /**
   * Query the nearest mesh surface point to a array of points.
   *
   * @param points Array of points to be queried. Float [Nx3] tensor.
   *
   * @return First, an array of the closest points found, float [Nx3]
   * tensor. Second, an array with the face indices, int64 [N] tensor.
   */
  std::pair<torch::Tensor, torch::Tensor> QueryClosest(
      const torch::Tensor &points);

  /**
   * Register it in Pybind.
   */
  static void RegisterPybind(pybind11::module &m);

 private:
  detail::AbstractOctNode *root_;
  torch::Tensor verts_;
};
}  // namespace slamtb
