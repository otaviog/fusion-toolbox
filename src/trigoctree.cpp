#include "trigoctree.hpp"

#include "metrics.hpp"

using namespace std;

namespace fiontb {

class ATrigOctNode {
 public:
  ATrigOctNode(const AABB &box) : box_(box) {}
  virtual ~ATrigOctNode() {}

  virtual pair<Eigen::Vector3f, long> QueryClosest(
      const Eigen::Vector3f bounded_qpoint,
      const Eigen::Vector3f &qpoint) const = 0;

  virtual pair<Eigen::Vector3f, long> QueryClosest(
      const Eigen::Vector3f &qpoint, float max_radius) const = 0;

  const AABB &get_box() const { return box_; }

 protected:
  AABB box_;
};

class LeafTrigOctNode : public ATrigOctNode {
 public:
  LeafTrigOctNode(const torch::Tensor &verts, torch::Tensor faces,
                  const AABB &box)
      : ATrigOctNode(box), verts_(verts), faces_(faces) {}

  pair<Eigen::Vector3f, long> QueryClosest(
      const Eigen::Vector3f /*bounded_qpoint*/,
      const Eigen::Vector3f &qpoint) const override {
    return QueryClosestPoint(qpoint, verts_, faces_);
  }

  pair<Eigen::Vector3f, long> QueryClosest(const Eigen::Vector3f &qpoint,
                                           float /*radius*/) const override {
    return QueryClosestPoint(qpoint, verts_, faces_);
  }

 private:
  const torch::Tensor &verts_;
  torch::Tensor faces_;
};

AABB CreateAABBFromFaces(const torch::Tensor &verts,
                         const torch::Tensor &faces) {
  const auto vacc = verts.accessor<float, 2>();
  const auto facc = faces.accessor<long, 2>();

  for (int face = 0; face < faces.size(0); ++face) {
  }
  return AABB();
}

class TrigOctNode : public ATrigOctNode {
 public:
  TrigOctNode(const torch::Tensor &verts, const torch::Tensor &faces,
              const AABB &split_box, int leaf_num_trigs)
      : ATrigOctNode(split_box) {
    AABB octo_boxes[8];
    SubdivideAABBOcto(box_, octo_boxes);

    const auto face_acc = faces.accessor<long, 2>();
    const auto vert_acc = verts.accessor<float, 2>();

    for (size_t box_idx = 0; box_idx < 8; ++box_idx) {
      const AABB &curr_box = octo_boxes[box_idx];
      children_[box_idx] = nullptr;

      torch::Tensor face_mask = torch::zeros({faces.size(0)}, torch::kUInt8);
      auto fmask_acc = face_mask.accessor<uint8_t, 1>();
      for (int face = 0; face < face_acc.size(0); ++face) {
        const long f0 = face_acc[face][0];
        const long f1 = face_acc[face][1];
        const long f2 = face_acc[face][2];

        const Eigen::Vector3f p0(vert_acc[f0][0], vert_acc[f0][1],
                                 vert_acc[f0][2]);
        const Eigen::Vector3f p1(vert_acc[f1][0], vert_acc[f1][1],
                                 vert_acc[f1][2]);
        const Eigen::Vector3f p2(vert_acc[f2][0], vert_acc[f2][1],
                                 vert_acc[f2][2]);

        if (curr_box.Intersects(p0, p1, p2)) {
          fmask_acc[face] = 1;
        }
      }

      torch::Tensor selected_faces =
          faces.masked_select(face_mask.view({-1, 1})).view({-1, 3});

      const int select_count = selected_faces.size(0);
      if (select_count == 0) continue;

      if (select_count < leaf_num_trigs) {
        children_[box_idx] =
            new LeafTrigOctNode(verts, selected_faces, curr_box);
      } else {
        children_[box_idx] =
            new TrigOctNode(verts, selected_faces, curr_box, leaf_num_trigs);
      }
    }
  }

  ~TrigOctNode() {
    for (int i = 0; i < 8; ++i) {
      delete children_[i];
    }
  }

  std::pair<Eigen::Vector3f, long> QueryClosest(
      const Eigen::Vector3f bounded_qpoint,
      const Eigen::Vector3f &qpoint) const override {
    pair<Eigen::Vector3f, long> result(Eigen::Vector3f(0, 0, 0), -1);
    const ATrigOctNode *visited = nullptr;
    for (int i = 0; i < 8; ++i) {
      const ATrigOctNode *child = children_[i];
      if (child == nullptr) {
        continue;
      }

      const AABB &box = child->get_box();
      if (box.IsInside(bounded_qpoint)) {
        visited = child;
        result = child->QueryClosest(bounded_qpoint, qpoint);
        break;
      }
    }

    float min_distance;
    if (result.second == -1) {
      min_distance = std::numeric_limits<float>::infinity();
    } else {
      min_distance = (result.first - qpoint).norm();
    }

    for (int i = 0; i < 8; ++i) {
      const ATrigOctNode *child = children_[i];
      if (child == nullptr || child == visited) {
        continue;
      }

      if (child->get_box().IsInside(qpoint, min_distance)) {
        const auto this_result = child->QueryClosest(qpoint, min_distance);

        if (this_result.second == -1) continue;

        const float distance = (this_result.first - qpoint).norm();
        if (distance < min_distance) {
          min_distance = distance;
          result = this_result;
        }
      }
    }
    return result;
  }

  std::pair<Eigen::Vector3f, long> QueryClosest(const Eigen::Vector3f &qpoint,
                                                float radius) const override {
    const float Inf = std::numeric_limits<float>::infinity();
    pair<Eigen::Vector3f, long> result(Eigen::Vector3f(Inf, Inf, Inf), -1);

    for (int i = 0; i < 8; ++i) {
      const ATrigOctNode *child = children_[i];
      if (child == nullptr) {
        continue;
      }

      if (child->get_box().IsInside(qpoint, radius)) {
        const auto this_result = child->QueryClosest(qpoint, radius);
        const float distance = (this_result.first - qpoint).norm();
        if (distance < radius) {
          radius = distance;
          result = this_result;
        }
      }
    }

    return result;
  }

 private:
  ATrigOctNode *children_[8];
};

TrigOctree::TrigOctree(torch::Tensor verts, const torch::Tensor &faces,
                       int leaf_num_trigs)
    : verts_(verts) {
  root_ = new TrigOctNode(verts_, faces, AABB(verts), leaf_num_trigs);
}

TrigOctree::~TrigOctree() { delete root_; }

pair<torch::Tensor, torch::Tensor> TrigOctree::QueryClosest(
    const torch::Tensor &points) {
  torch::Tensor closest_mtx = torch::empty({points.size(0), 3}, torch::kFloat);

  torch::Tensor idx_mtx = torch::empty({points.size(0)}, torch::kInt64);

  auto cls_acc = closest_mtx.accessor<float, 2>();
  auto idx_acc = idx_mtx.accessor<long, 1>();
  const auto qacc = points.accessor<float, 2>();

  for (int i = 0; i < points.size(0); ++i) {
    Eigen::Vector3f qpoint(qacc[i][0], qacc[i][1], qacc[i][2]);
    const auto bounded_qpoint = root_->get_box().GetClosestPoint(qpoint);

    const auto result = root_->QueryClosest(bounded_qpoint, qpoint);
    cls_acc[i][0] = result.first[0];
    cls_acc[i][1] = result.first[1];
    cls_acc[i][2] = result.first[2];

    idx_acc[i] = result.second;
  }

  return make_pair(closest_mtx, idx_mtx);
}

}  // namespace fiontb
