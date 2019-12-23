#include "slamfeat.hpp"

#include <random>

#include <torch/csrc/utils/pybind.h>

#include "camera.hpp"

namespace fiontb {
static int rand_int(int start, int end) {
  return start + std::rand() % (end - start);
}

static int rand_int_except(int start, int end, int except) {
  while (true) {
    const int val = rand_int(start, end);
    if (val != except) return val;
  }
}

static bool dice(float chance) {
  const double roll = double(rand() % 1000) / 1000.0;
  return roll < chance;
}

int SlamFeatOp::ExtractPatch(
    const torch::Tensor &anc_points_, const torch::Tensor &anc_colors_,
    const torch::Tensor &anc_mask_, const torch::Tensor &anc_cam_to_world_,
    const torch::Tensor &pos_points_, const torch::Tensor &pos_colors_,
    const torch::Tensor &pos_mask_, const torch::Tensor &pos_world_to_cam_,
    const torch::Tensor &pos_cam_to_world_, const torch::Tensor &pos_kcam_,
    const torch::Tensor &neg_colors_, float point_dist_thresh,
    torch::Tensor anc_patch_, torch::Tensor anc_d_patch_,
    torch::Tensor pos_patch_, torch::Tensor pos_d_patch_, torch::Tensor mask_,
    torch::Tensor neg_patch_, torch::Tensor neg_d_patch_,
    torch::Tensor hard_negative_) {
  const auto anc_points = anc_points_.accessor<float, 3>();
  const auto anc_colors = anc_colors_.accessor<unsigned char, 3>();
  const auto anc_mask = anc_mask_.accessor<bool, 2>();

  const auto pos_points = pos_points_.accessor<float, 3>();
  const auto pos_colors = pos_colors_.accessor<unsigned char, 3>();

  const auto pos_mask = pos_mask_.accessor<bool, 2>();
  const auto neg_colors = neg_colors_.accessor<unsigned char, 3>();

  const RigidTransform<kCPU, float> anc_cam_to_world(anc_cam_to_world_);
  const RigidTransform<kCPU, float> pos_world_to_cam(pos_world_to_cam_);
  const RigidTransform<kCPU, float> pos_cam_to_world(pos_cam_to_world_);
  const KCamera<kCPU, float> pos_kcam(pos_kcam_);

  auto anc_patch = anc_patch_.accessor<unsigned char, 3>();
  auto anc_d_patch = anc_d_patch_.accessor<float, 2>();
  auto pos_patch = pos_patch_.accessor<unsigned char, 3>();
  auto pos_d_patch = pos_d_patch_.accessor<float, 2>();
  auto neg_patch = neg_patch_.accessor<unsigned char, 3>();
  auto neg_d_patch = neg_d_patch_.accessor<float, 2>();

  auto mask = mask_.accessor<bool, 2>();
  auto hard_negative = hard_negative_.accessor<bool, 2>();

  int mask_count = 0;
  for (int row = 0; row < anc_points.size(0); ++row) {
    for (int col = 0; col < anc_points.size(1); ++col) {
      mask[row][col] = false;

      if (!anc_mask[row][col]) {
        continue;
      }

      const Eigen::Vector3f anc_point(to_vec3<float>(anc_points[row][col]));
      const Eigen::Vector3f world_point(anc_cam_to_world.Transform(anc_point));
      const Eigen::Vector3f pred_pos_point(
          pos_world_to_cam.Transform(world_point));
      int u, v;
      pos_kcam.Projecti(pred_pos_point, u, v);

      if (u < 0 || u > pos_points.size(1) || v < 0 || v > pos_points.size(0)) {
        continue;
      }

      if (!pos_mask[row][col]) {
        continue;
      }

      const Eigen::Vector3f pos_point(
          pos_cam_to_world.Transform(to_vec3<float>(pos_points[v][u])));
      if ((pos_point - world_point).squaredNorm() > point_dist_thresh) {
        continue;
      }

      for (int k = 0; k < 3; ++k) {
        anc_patch[row][col][k] = anc_colors[row][col][k];
        pos_patch[row][col][k] = pos_colors[v][u][k];
      }
      anc_d_patch[row][col] = anc_points[row][col][2];
      pos_d_patch[row][col] = pos_points[v][u][2];

      mask[row][col] = true;
      ++mask_count;

      if (dice(0.5)) {
        const int neg_u = rand_int_except(
            std::max(u - 5, 0), std::min(u + 5, int(pos_colors.size(1))), u);
        const int neg_v = rand_int_except(
            std::max(v - 5, 0), std::min(v + 5, int(pos_colors.size(0))), v);

        for (int k = 0; k < 3; ++k) {
          neg_patch[row][col][k] = pos_colors[neg_v][neg_u][k];
        }
        neg_d_patch[row][col] = pos_points[neg_v][neg_u][2];
        hard_negative[row][col] = true;
      } else {
        const int neg_u = rand_int(0, neg_colors.size(1));
        const int neg_v = rand_int(0, neg_colors.size(0));

        for (int k = 0; k < 3; ++k) {
          neg_patch[row][col][k] = neg_colors[neg_v][neg_u][k];
        }
        neg_d_patch[row][col] = (float(rand() % 1000) / 1000.0) * 3.0f;
        hard_negative[row][col] = false;
      }
    }
  }

  return mask_count;
}

void SlamFeatOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<SlamFeatOp>(m, "SlamFeatOp")
      .def_static("extract_patch", &SlamFeatOp::ExtractPatch);
}
}  // namespace fiontb
