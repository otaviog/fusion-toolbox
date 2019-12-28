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

struct SFFrameAccessor {
  torch::TensorAccessor<bool, 2> mask;
  torch::TensorAccessor<float, 3> points;
  torch::TensorAccessor<uint8_t, 3> colors;
  torch::TensorAccessor<int32_t, 2> depths;

  SFFrameAccessor(SFFrame pcl)
      : mask(pcl.mask.accessor<bool, 2>()),
        points(pcl.points.accessor<float, 3>()),
        colors(pcl.colors.accessor<uint8_t, 3>()),
        depths(pcl.depths.accessor<int32_t, 2>()) {}

  int get_width() const { return points.size(1); }

  int get_height() const { return points.size(0); }

  bool is_valid(int row, int col) const {
    if (row < 0 || row >= get_height() || col < 0 || col >= get_width())
      return false;

    return mask[row][col];
  }

  Eigen::Vector3f get_point(int row, int col) const {
    return to_vec3<float>(points[row][col]);
  }

  Eigen::Vector3i get_color(int row, int col) const {
    return to_vec3<int>(colors[row][col]);
  }

  int get_depth(int row, int col) const { return depths[row][col]; }
};

struct SFCropAccessor {
  torch::TensorAccessor<uint8_t, 3> colors;
  torch::TensorAccessor<int32_t, 2> depths;

  SFCropAccessor(SFCrop crop)
      : colors(crop.colors.accessor<uint8_t, 3>()),
        depths(crop.depths.accessor<int32_t, 2>()) {}

  void set_color(int row, int col, const Eigen::Vector3i &color) {
    colors[row][col][0] = color[0];
    colors[row][col][1] = color[1];
    colors[row][col][2] = color[2];
  }

  void set_depth(int row, int col, int depth) { depths[row][col] = depth; }
};

namespace {
int GenerateTripletImpl(const SFFrameAccessor &anch_frame,
                        const RigidTransform<kCPU, float> &anch_cam_to_world,
                        const SFFrameAccessor &posv_frame,
                        const KCamera<kCPU, float> posv_kcam,
                        const RigidTransform<kCPU, float> &posv_cam_to_world,
                        const RigidTransform<kCPU, float> &posv_world_to_cam,
                        float point_dist_thresh,
                        const SFFrameAccessor &negv_frame,
                        SFCropAccessor anch_crop, SFCropAccessor posv_crop,
                        torch::TensorAccessor<bool, 2> posv_mask,
                        SFCropAccessor negv_crop,
                        torch::TensorAccessor<bool, 2> negv_mask,
                        torch::TensorAccessor<bool, 2> negv_hard) {
  bool is_hard_negv = dice(0.5);
  const int hn_u = rand_int_except(-10, 10, 0);
  const int hn_v = rand_int_except(-10, 10, 0);

  int mask_count;
  for (int row = 0; row < anch_frame.get_height(); ++row) {
    for (int col = 0; col < anch_frame.get_width(); ++col) {
      posv_mask[row][col] = false;
      negv_mask[row][col] = false;
      negv_hard[row][col] = false;

      anch_crop.set_color(row, col, anch_frame.get_color(row, col));
      anch_crop.set_depth(row, col, anch_frame.get_depth(row, col));

      if (!anch_frame.is_valid(row, col)) {
        continue;
      }

      const Eigen::Vector3f anc_point(anch_frame.get_point(row, col));
      const Eigen::Vector3f world_point(anch_cam_to_world.Transform(anc_point));
      const Eigen::Vector3f pred_pos_point(
          posv_world_to_cam.Transform(world_point));
      int u, v;
      posv_kcam.Projecti(pred_pos_point, u, v);
      if (u < 0 || u > posv_frame.get_width() || v < 0 ||
          v > posv_frame.get_height()) {
        continue;
      }

      posv_crop.set_color(row, col, posv_frame.get_color(v, u));
      posv_crop.set_depth(row, col, posv_frame.get_depth(v, u));

      const Eigen::Vector3f pos_point(
          posv_cam_to_world.Transform(posv_frame.get_point(v, u)));
      if ((pos_point - world_point).squaredNorm() <= point_dist_thresh) {
        if (anch_frame.is_valid(row, col) && posv_frame.is_valid(v, u)) {
          posv_mask[row][col] = true;
          ++mask_count;
        }
      }

      if (is_hard_negv) {
        const int neg_u = u + hn_u;
        const int neg_v = v + hn_v;
        if (posv_frame.is_valid(neg_v, neg_u)) {
          negv_crop.set_color(row, col, posv_frame.get_color(neg_v, neg_u));
          negv_crop.set_depth(row, col, posv_frame.get_depth(neg_v, neg_u));
          negv_mask[row][col] = true;
          negv_hard[row][col] = true;
        }
      } else {
        negv_crop.set_color(row, col, negv_frame.get_color(row, col));
        negv_crop.set_depth(row, col, negv_frame.get_depth(row, col));
        negv_mask[row][col] = negv_frame.is_valid(row, col);
      }
    }
  }
  return mask_count;
}

}  // namespace

int SlamFeatOp::GenerateTriplet(
    const SFFrame &anch_frame, const torch::Tensor anch_cam_to_world,
    const SFFrame &posv_frame, const torch::Tensor &posv_kcam,
    const torch::Tensor posv_cam_to_world, float point_dist_thresh,
    const SFFrame &negv_frame, SFCrop anch_crop, SFCrop posv_crop,
    torch::Tensor posv_mask, SFCrop negv_crop, torch::Tensor negv_mask,
    torch::Tensor negv_hard) {
  return GenerateTripletImpl(
      SFFrameAccessor(anch_frame),
      RigidTransform<kCPU, float>(anch_cam_to_world),
      SFFrameAccessor(posv_frame), KCamera<kCPU, float>(posv_kcam),
      RigidTransform<kCPU, float>(posv_cam_to_world),
      RigidTransform<kCPU, float>(posv_cam_to_world.inverse()),
      point_dist_thresh, SFFrameAccessor(negv_frame), SFCropAccessor(anch_crop),
      SFCropAccessor(posv_crop), posv_mask.accessor<bool, 2>(),
      SFCropAccessor(negv_crop), negv_mask.accessor<bool, 2>(),
      negv_hard.accessor<bool, 2>());
}

void SlamFeatOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<SlamFeatOp>(m, "SlamFeatOp")
      .def_static("extract_patch", &SlamFeatOp::ExtractPatch)
      .def_static("generate_triplet", &SlamFeatOp::GenerateTriplet);

  pybind11::class_<SFFrame>(m, "SFFrame")
      .def(pybind11::init())
      .def_readwrite("mask", &SFFrame::mask)
      .def_readwrite("points", &SFFrame::points)
      .def_readwrite("colors", &SFFrame::colors)
      .def_readwrite("depths", &SFFrame::depths);

  pybind11::class_<SFCrop>(m, "SFCrop")
      .def(pybind11::init())
      .def_readwrite("colors", &SFCrop::colors)
      .def_readwrite("depths", &SFCrop::depths);
}
}  // namespace fiontb
