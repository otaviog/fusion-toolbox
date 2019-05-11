#include "octree.hpp"

#include <map>
#include <vector>

#include <ATen/core/Formatting.h>

#include "cuda_runtime_api.h"

using namespace std;

namespace fiontb {
// Defined in .cu
torch::Tensor GPUPointDistances(const torch::Tensor &qpoints_idxs,
                                const torch::Tensor &qpoints,
                                const torch::Tensor &mpoints_idxs,
                                const torch::Tensor &mpoints);

torch::Tensor GPUPointDistances(int qidx, float qx, float qy, float qz,
                                const torch::Tensor &mpoints_idxs,
                                const torch::Tensor &mpoints);

void GPUCopyResultToTensor(const torch::Tensor &sorted_dists,
                           const torch::Tensor &sorted_indices,
                           const torch::Tensor &indices, int qidx, float radius,
                           torch::Tensor out_distances,
                           torch::Tensor out_indices);

namespace { 
static void CPUCopyResultToTensor(
    const torch::TensorAccessor<float, 1> &sorted_dists,
    const torch::TensorAccessor<long, 1> &indices,
    const torch::TensorAccessor<long, 1> &sorted_indices, const int qpoint,
    float radius, torch::TensorAccessor<float, 2> out_distances,
    torch::TensorAccessor<long, 2> out_indices) {
  for (int i = 0; i < min(sorted_dists.size(0), out_distances.size(1)); ++i) {
    const float dist = sorted_dists[i];
    if (dist > radius) {
      break;
    }

    out_distances[qpoint][i] = dist;
    out_indices[qpoint][i] = indices[sorted_indices[i]];
  }
}
}

namespace priv {

class OctNode {
 public:
  OctNode(const torch::Tensor &indices, torch::Tensor points, const AABB &box,
          int leaf_num_points);
  OctNode(torch::Tensor indices, torch::Tensor points, const AABB &box);
  OctNode(const OctNode &) = delete;
  OctNode &operator=(OctNode const &) = delete;
  ~OctNode();

  const OctNode *get_child(int idx) const { return children_[idx]; }

  const AABB &get_box() const { return box_; }

  AABB get_box() { return box_; }

  bool IsLeaf() const { return leaf_; }

  const torch::Tensor get_indices() const { return indices_; }

  void Query(const Eigen::Vector3f qpoint, int max_k, float radius,
             std::vector<const OctNode *> &leafs) const;

  int GetLeafNodeCount() const;

 private:
  bool leaf_;
  OctNode *children_[8];
  torch::Tensor indices_;
  AABB box_;
};

OctNode::OctNode(const torch::Tensor &indices, torch::Tensor points,
                 const AABB &split_box, int leaf_num_points)
    : leaf_(false) {
  using Eigen::Vector3f;

  box_ = AABB(points.index_select(0, indices));
  // box_ = split_box;
  const Vector3f mi = box_.get_min();
  const Vector3f ma = box_.get_max();
  const Vector3f ce = (mi + ma) * 0.5;

  const AABB boxes[] = {AABB(Vector3f(mi[0], mi[1], mi[2]), ce),
                        AABB(Vector3f(mi[0], mi[1], ma[2]), ce),
                        AABB(Vector3f(mi[0], ma[1], ma[2]), ce),
                        AABB(Vector3f(mi[0], ma[1], mi[2]), ce),
                        AABB(Vector3f(ma[0], mi[1], mi[2]), ce),
                        AABB(Vector3f(ma[0], mi[1], ma[2]), ce),
                        AABB(Vector3f(ma[0], ma[1], ma[2]), ce),
                        AABB(Vector3f(ma[0], ma[1], mi[2]), ce)};
  for (int box_idx = 0; box_idx < 8; ++box_idx) {
    const AABB &curr_box = boxes[box_idx];
    children_[box_idx] = nullptr;

    torch::Tensor mask = curr_box.IsInside(indices, points);
    torch::Tensor inside_indices = indices.masked_select(mask);
    const int inside_count = inside_indices.size(0);
    if (inside_count == 0) {
      continue;
    }
    if (inside_count < leaf_num_points) {
      children_[box_idx] = new OctNode(inside_indices, points, curr_box);
    } else {
      children_[box_idx] =
          new OctNode(inside_indices, points, curr_box, leaf_num_points);
    }
  }
}

OctNode::OctNode(torch::Tensor indices, torch::Tensor points, const AABB &box)
    : leaf_(true), indices_(indices), box_(box) {

  for (int box_idx = 0; box_idx < 8; ++box_idx) {
    children_[box_idx] = nullptr;
  }
}

OctNode::~OctNode() {
  for (int b = 0; b < 8; ++b) {
    delete children_[b];
  }
}

void dump_tensor(torch::Tensor tensor, const string &out) {
  std::ofstream file(out.c_str());
  at::print(file, tensor, 99);
}

void OctNode::Query(const Eigen::Vector3f qpoint, int max_k, float radius,
                    std::vector<const OctNode *> &leafs) const {
  if (IsLeaf()) {
    leafs.push_back(this);
    return;
  }

  for (int i = 0; i < 8; ++i) {
    const OctNode *child = children_[i];
    if (child == nullptr) {
      continue;
    }

    const AABB &box = child->get_box();
    if (!box.IsInside(qpoint, radius)) {
      continue;
    }

    child->Query(qpoint, max_k, radius, leafs);
  }
}

int OctNode::GetLeafNodeCount() const {
  if (IsLeaf()) {
    return 1;
  }

  int count = 0;
  for (int i = 0; i < 8; ++i) {
    if (children_[i] == nullptr) continue;
    count += children_[i]->GetLeafNodeCount();
  }

  return count;
}
}  // namespace priv

Octree::Octree(torch::Tensor points, int leaf_num_points)
    : box_(AABB(points)), points_(points) {
  torch::TensorOptions opts(torch::kInt64);
  opts = opts.device(points.device());
  root_ = new priv::OctNode(torch::arange(0, points.size(0), opts).squeeze(),
                            points, box_, leaf_num_points);
}

Octree::~Octree() { delete root_; }

std::pair<torch::Tensor, torch::Tensor> Octree::Query(
    const torch::Tensor _qpoints, int max_k, float radius) {
  torch::Tensor qpoints_cpu;
  if (_qpoints.is_cuda()) {
    qpoints_cpu = _qpoints.cpu();
  } else {
    qpoints_cpu = _qpoints;
  }

  torch::Tensor dist_mtx =
      torch::full({_qpoints.size(0), max_k},
                  std::numeric_limits<float>::infinity(), points_.device());

  torch::Tensor idx_mtx =
      torch::full({_qpoints.size(0), max_k}, -1,
                  torch::TensorOptions(torch::kInt64).device(points_.device()));

  const auto qcc = qpoints_cpu.accessor<float, 2>();
  vector<const priv::OctNode *> leafs;
  leafs.reserve(10);
  for (int i = 0; i < qcc.size(0); ++i) {
    Eigen::Vector3f qpoint(qcc[i][0], qcc[i][1], qcc[i][2]);
    leafs.clear();

    root_->Query(qpoint, max_k, radius, leafs);

    if (leafs.empty()) {
      continue;
    }
    vector<torch::Tensor> idx_vec;
    idx_vec.reserve(leafs.size());
    for (size_t j = 0; j < leafs.size(); ++j) {
      idx_vec.push_back(leafs[j]->get_indices());
    }

    torch::Tensor indices = torch::cat(idx_vec);
    torch::Tensor distances =
        GPUPointDistances(i, qpoint[0], qpoint[1], qpoint[2], indices, points_);

    auto sorting =
        distances.topk(std::min(long(max_k), distances.size(1)), 1, false);

    auto sorted_dists = std::get<0>(sorting);
    auto sorted_idxs = std::get<1>(sorting);

    GPUCopyResultToTensor(sorted_dists, sorted_idxs, indices, i, radius,
                          dist_mtx, idx_mtx);
  }

  return make_pair(dist_mtx, idx_mtx);
}

int Octree::get_leaf_node_count() const { return root_->GetLeafNodeCount(); }
}  // namespace fiontb
