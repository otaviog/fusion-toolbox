#include "geometry.hpp"

#include <cmath>

using namespace std;

namespace fiontb {

namespace {
bool IsPointInsideTriangle(const Eigen::Vector3f &query,
                           const Eigen::Vector3f &p0, const Eigen::Vector3f &p1,
                           const Eigen::Vector3f &p2) {
  const Eigen::Vector3f q0 = p0 - query;
  const Eigen::Vector3f q1 = p1 - query;
  const Eigen::Vector3f q2 = p2 - query;

  const Eigen::Vector3f norm_12 = q1.cross(q2);
  const Eigen::Vector3f norm_20 = q2.cross(q0);
  const Eigen::Vector3f norm_01 = q0.cross(q1);

  if (norm_12.dot(norm_20) < 0.0f) {
    return false;
  } else if (norm_12.dot(norm_01) < 0.0f) {
    return false;
  }

  return true;
}

Eigen::Vector3f GetClosestPointLine(const Eigen::Vector3f &query,
                                    const Eigen::Vector3f &p0,
                                    const Eigen::Vector3f &p1) {
  Eigen::Vector3f vec = p1 - p0;
  float t = (query - p0).dot(vec) / vec.dot(vec);
  t = min(max(t, 0.0f), 1.0f);

  return p0 + vec * t;
}

Eigen::Vector3f GetClosestPointTriangle(const Eigen::Vector3f &query,
                                        const Eigen::Vector3f &p0,
                                        const Eigen::Vector3f &p1,
                                        const Eigen::Vector3f &p2) {
  const Eigen::Vector3f plane_normal = (p1 - p0).cross(p2 - p0).normalized();
  const float plane_dist = plane_normal.dot(p0);

  const Eigen::Vector3f closest =
      query - plane_normal * (plane_normal.dot(query) - plane_dist);

  if (IsPointInsideTriangle(closest, p0, p1, p2)) {
    return closest;
  }

  const Eigen::Vector3f c0 = GetClosestPointLine(query, p0, p1);
  const Eigen::Vector3f c1 = GetClosestPointLine(query, p1, p2);
  const Eigen::Vector3f c2 = GetClosestPointLine(query, p2, p0);

  const float d0 = (query - c0).squaredNorm();
  const float d1 = (query - c1).squaredNorm();
  const float d2 = (query - c2).squaredNorm();

  if (d0 < d1 && d0 < d2) {
    return c0;
  } else if (d1 < d0 && d1 < d2) {
    return c1;
  }

  return c2;
}
}  // namespace

pair<Eigen::Vector3f, long> GetClosestPoint(const Eigen::Vector3f &qpoint,
                                            const torch::Tensor &verts,
                                            const torch::Tensor &faces) {
  float min_distance = std::numeric_limits<float>::infinity();
  Eigen::Vector3f min_closest;
  long min_face;

  const auto face_acc = faces.accessor<long, 2>();
  const auto vert_acc = verts.accessor<float, 2>();

  for (long i = 0; i < face_acc.size(0); ++i) {
    const long f0 = face_acc[i][0];
    const long f1 = face_acc[i][1];
    const long f2 = face_acc[i][2];

    const Eigen::Vector3f p0(vert_acc[f0][0], vert_acc[f0][1], vert_acc[f0][2]);
    const Eigen::Vector3f p1(vert_acc[f1][0], vert_acc[f1][1], vert_acc[f1][2]);
    const Eigen::Vector3f p2(vert_acc[f2][0], vert_acc[f2][1], vert_acc[f2][2]);

    const Eigen::Vector3f closest = GetClosestPointTriangle(qpoint, p0, p1, p2);
    const float distance = (closest - qpoint).squaredNorm();

    if (distance < min_distance) {
      min_distance = distance;
      min_closest = closest;
      min_face = i;
    }
  }

  return make_pair(min_closest, min_face);
}

}  // namespace fiontb
