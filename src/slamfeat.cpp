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
