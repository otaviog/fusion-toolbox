#include "octtree.hpp"

#include <map>
#include <vector>

#include <ATen/core/Formatting.h>

using namespace std;

namespace fiontb {
namespace priv {
struct QueryResult {
  std::vector<torch::Tensor> distances, index;
};

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

  void Query(torch::Tensor qpoint, torch::Tensor which_qpoints, float radius,
             std::map<long, QueryResult> &result) const;

 private:
  bool leaf_;
  OctNode *children_[8];
  torch::Tensor indices_;
  torch::Tensor points_;
  AABB box_;
};

OctNode::OctNode(const torch::Tensor &indices, torch::Tensor points,
                 const AABB &azdbox, int leaf_num_points)
    : leaf_(false) {
  using Eigen::Vector3f;

  // box_ = AABB(points.index_select(0, indices));
  box_ = azdbox;
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

    const int inside_count = mask.sum().item<int>();
    if (inside_count == 0) continue;

    torch::Tensor inside_indices = indices.masked_select(mask);
    if (inside_count < leaf_num_points) {
      children_[box_idx] = new OctNode(inside_indices, points, curr_box);
    } else {
      children_[box_idx] =
          new OctNode(inside_indices, points, curr_box, leaf_num_points);
    }
  }
}

OctNode::OctNode(torch::Tensor indices, torch::Tensor points, const AABB &box)
    : leaf_(true), indices_(indices), points_(points), box_(box) {
  for (int box_idx = 0; box_idx < 8; ++box_idx) {
    children_[box_idx] = nullptr;
  }
}

OctNode::~OctNode() {
  for (int b = 0; b < 8; ++b) {
    delete children_[b];
  }
}

void OctNode::Query(torch::Tensor qpoints, torch::Tensor which_qpoints,
                    float radius, map<long, QueryResult> &results) const {
  if (IsLeaf()) {
    auto local_mpoints = points_.index_select(0, indices_);

    torch::Tensor distances = torch::cdist(local_mpoints, qpoints);
    auto wq_acc = which_qpoints.accessor<long, 1>();

    for (long i = 0; i < which_qpoints.size(0); ++i) {
      const long idx = wq_acc[i];
      results[idx].distances.push_back(distances.narrow(1, i, 1));
      results[idx].index.push_back(indices_);
    }

    return;
  }

  for (int i = 0; i < 8; ++i) {
    const OctNode *child = children_[i];
    if (child == nullptr) {
      continue;
    }

    const AABB &box = child->get_box();
    auto mask = box.IsInside(qpoints, radius);

    const int inside_count = mask.sum().item<int>();
    if (inside_count == 0) {
      continue;
    }

    torch::Tensor sub_qpoints =
        qpoints.masked_select(mask.view({-1, 1})).view({-1, 3});
    torch::Tensor sub_which_qpoints = which_qpoints.masked_select(mask.cpu());
    child->Query(sub_qpoints, sub_which_qpoints, radius, results);
  }
}
}  // namespace priv

OctTree::OctTree(torch::Tensor points, int leaf_num_points)
    : box_(AABB(points)) {
  torch::TensorOptions opts(torch::kInt64);
  opts = opts.device(points.device());
  root_ = new priv::OctNode(torch::arange(0, points.size(0), opts).squeeze(),
                            points, box_, leaf_num_points);
}

OctTree::~OctTree() { delete root_; }

void CopyResultToTensor(const torch::TensorAccessor<float, 1> &sorted_dists,
                        const torch::TensorAccessor<long, 1> &indices,
                        const torch::TensorAccessor<long, 1> &sorted_indices,
                        const int qpoint, float radius,
                        torch::TensorAccessor<float, 2> out_distances,
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

pair<torch::Tensor, torch::Tensor> OctTree::Query(const torch::Tensor &qpoints,
                                                  int max_k, float radius) {
  if (max_k == 1000) {
    at::print(qpoints, 99);
  }
  torch::Tensor which_qpoints =
      torch::arange(0, qpoints.size(0), torch::kInt64).squeeze();
  map<long, priv::QueryResult> results;
  auto iqpoints = root_->get_box().GetClosestPoint(qpoints);
  root_->Query(iqpoints, which_qpoints, radius, results);

  torch::Tensor dist_mtx = torch::full({qpoints.size(0), max_k},
                                       std::numeric_limits<float>::infinity());
  torch::Tensor idx_mtx =
      torch::full({qpoints.size(0), max_k}, -1, torch::kInt64);

  for (auto res : results) {
    const long qpoint_idx = res.first;
    const priv::QueryResult &qres = res.second;

    auto distances = torch::cat(qres.distances);
    auto indices = torch::cat(qres.index).cpu().view(-1);

    auto sorting = distances.sort(0);
    auto sorted_distances = std::get<0>(sorting).cpu().view(-1);
    auto sorted_indices = std::get<1>(sorting).cpu().view(-1);

    auto a1 = sorted_distances.accessor<float, 1>();
    auto a2 = indices.accessor<long, 1>();
    auto a3 = sorted_indices.accessor<long, 1>();
    auto a4 = dist_mtx.accessor<float, 2>();
    auto a5 = idx_mtx.accessor<long, 2>();

    CopyResultToTensor(a1, a2, a3, qpoint_idx, radius, a4, a5);
  }

  return make_pair(dist_mtx, idx_mtx);
}


}  // namespace fiontb
