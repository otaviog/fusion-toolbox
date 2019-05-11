#pragma once

#include <torch/torch.h>

#include "aabb.hpp"

namespace fiontb {
namespace priv {
class OctNode;
}

/**
 * Octree search of k-nearest neighbors points. It's implemented on
 * PyTorch for either GPU or CPU tensors.
 */
class Octree {
 public:
  /**
   * Construct a Octree for searching a given set of points.
   *
   * @param points [Nx3] tensor of search target 3D points. Expected
   * floating point tensor type.
   * 
   * @param leaf_num_points number of points in the leaf nodes. Large
   * numbers between 512-2048 are recommended.
   */
  Octree(torch::Tensor points, int leaf_num_points = 128);
  Octree(const Octree &) = delete;
  Octree &operator=(Octree const &) = delete;

  ~Octree();

  /**
   * Search for the k nearest points within a given radius of
   * distance.
   *
   * @param qpoints [Mx3] query points.
   * @param max_k max_k neighbors.
   * @param radius the maximum distance between points.
   *
   * @return  The first tensor is the [MxK] matrix of distances
   * between query points to the K nearest ones in the octree. If not
   * enough points are within the argument radius, then remaining
   * elements on the row are filled with `inf`. The second tensor is
   * [MxK] matrix of row-indices to the instance's input point tensor
   * in the same order as the result of the distances. Entries valued
   * with `inf` in the first tensor are marked with `-1` in the
   * second.
   */
  std::pair<torch::Tensor, torch::Tensor> Query(torch::Tensor qpoints,
                                                int max_k, float radius);

  int get_leaf_node_count() const;
  
 private:
  AABB box_;
  priv::OctNode *root_;
  torch::Tensor points_;
};
}  // namespace fiontb
