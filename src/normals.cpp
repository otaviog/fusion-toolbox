#include "normals.hpp"

#include <iostream>

#include "eigen_common.hpp"

namespace fiontb {

void ComputeCentralDifferences_gpu(const torch::Tensor xyz_image,
                                   const torch::Tensor mask_image,
                                   torch::Tensor normals);

namespace {
inline Eigen::Vector3f GetNormal(const Eigen::Vector3f &p0,
                                 const Eigen::Vector3f &p1,
                                 const Eigen::Vector3f &p2) {
  return (p1 - p0).cross(p2 - p0).normalized();
}

void ComputeAverage8(const torch::Tensor xyz_image,
                     const torch::Tensor mask_image,
                     torch::Tensor out_normals) {
  torch::Tensor normal_image =
      torch::empty({xyz_image.size(0), xyz_image.size(1), 3}, torch::kFloat);
  auto n_acc = normal_image.accessor<float, 3>();

  const auto xyz_a = xyz_image.accessor<float, 3>();
  const auto mask_a = mask_image.accessor<uint8_t, 2>();

  const int where[][2] = {{0, 1},  {1, 1},   {1, 0},  {1, -1},
                          {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

  const int iwidth = xyz_image.size(1);
  const int iheight = xyz_image.size(0);

  for (int row = 0; row < xyz_image.size(0); ++row) {
    for (int col = 0; col < xyz_image.size(1); ++col) {
      n_acc[row][col][0] = n_acc[row][col][1] = n_acc[row][col][2] = 0.0f;

      if (mask_a[row][col] == 0) continue;

      Eigen::Vector3f center(xyz_a[row][col][0], xyz_a[row][col][1],
                             xyz_a[row][col][2]);
      Eigen::Vector3f norm(0, 0, 0);
      int count = 0;

      for (int i = 0; i < 8; ++i) {
        const int r1 = row + where[i][1];
        if (r1 < 0 || r1 >= iheight) continue;

        const int c1 = col + where[i][0];
        if (c1 < 0 || c1 >= iwidth) continue;

        if (mask_a[r1][c1] == 0) continue;

        const int j = (i + 2) % 8;
        const int r2 = row + where[j][1];
        if (r2 < 0 || r2 >= iheight) continue;

        const int c2 = col + where[j][0];
        if (c2 < 0 || c2 >= iwidth) continue;

        if (mask_a[r2][c2] == 0) continue;

        norm += GetNormal(center,
                          Eigen::Vector3f(xyz_a[r2][c2][0], xyz_a[r2][c2][1],
                                          xyz_a[r2][c2][2]),
                          Eigen::Vector3f(xyz_a[r1][c1][0], xyz_a[r1][c1][1],
                                          xyz_a[r1][c1][2]));
        ++count;
      }
      if (count > 0) {
        norm = norm / count;
        norm.normalize();
        n_acc[row][col][0] = norm[0];
        n_acc[row][col][1] = norm[1];
        n_acc[row][col][2] = norm[2];
      }
    }
  }
}

template <typename scalar_t>
void ComputeCentralDifferences_cpu_kernel(const torch::Tensor xyz_image,
                                          const torch::Tensor mask_image,
                                          torch::Tensor out_normals) {
  auto n_acc = out_normals.accessor<scalar_t, 3>();

  const auto xyz_acc = xyz_image.accessor<scalar_t, 3>();
  const auto mask_acc = mask_image.accessor<uint8_t, 2>();

  const int iwidth = xyz_image.size(1);
  const int iheight = xyz_image.size(0);

  for (int row = 0; row < xyz_image.size(0); ++row) {
    for (int col = 0; col < xyz_image.size(1); ++col) {
      n_acc[row][col][0] = n_acc[row][col][1] = n_acc[row][col][2] = 0.0f;

      if (mask_acc[row][col] == 0) continue;

      const Eigen::Vector3f center(xyz_acc[row][col][0], xyz_acc[row][col][1],
                                   xyz_acc[row][col][2]);

      Eigen::Vector3f left = Eigen::Vector3f::Zero();
      if (col > 0 && mask_acc[row][col - 1] == 1) {
        left =
            Eigen::Vector3f(xyz_acc[row][col - 1][0], xyz_acc[row][col - 1][1],
                            xyz_acc[row][col - 1][2]);
      }

      Eigen::Vector3f right = Eigen::Vector3f::Zero();
      if (col < iwidth - 1 && mask_acc[row][col + 1] == 1) {
        right =
            Eigen::Vector3f(xyz_acc[row][col + 1][0], xyz_acc[row][col + 1][1],
                            xyz_acc[row][col + 1][2]);
      }

      Eigen::Vector3f top = Eigen::Vector3f::Zero();
      if (row > 0 && mask_acc[row - 1][col] == 1) {
        top =
            Eigen::Vector3f(xyz_acc[row - 1][col][0], xyz_acc[row - 1][col][1],
                            xyz_acc[row - 1][col][2]);
      }

      Eigen::Vector3f bottom = Eigen::Vector3f::Zero();
      if (row < iheight - 1 && mask_acc[row + 1][col] == 1) {
        bottom =
            Eigen::Vector3f(xyz_acc[row + 1][col][0], xyz_acc[row + 1][col][1],
                            xyz_acc[row + 1][col][2]);
      }

      constexpr float kRatioThreshold = 2.f;
      constexpr float kRatioThresholdSquared =
          kRatioThreshold * kRatioThreshold;

      float left_dist_squared = (left - center).squaredNorm();
      float right_dist_squared = (right - center).squaredNorm();
      float left_right_ratio = left_dist_squared / right_dist_squared;

      Eigen::Vector3f left_to_right;
      if (left_right_ratio < kRatioThresholdSquared &&
          left_right_ratio > 1.f / kRatioThresholdSquared) {
        left_to_right = right - left;
      } else if (left_dist_squared < right_dist_squared) {
        left_to_right = center - left;
      } else {  // left_dist_squared >= right_dist_squared
        left_to_right = right - center;
      }

      float bottom_dist_squared = (bottom - center).squaredNorm();
      float top_dist_squared = (top - center).squaredNorm();
      float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
      Eigen::Vector3f bottom_to_top;
      if (bottom_top_ratio < kRatioThresholdSquared &&
          bottom_top_ratio > 1.f / kRatioThresholdSquared) {
        bottom_to_top = top - bottom;
      } else if (bottom_dist_squared < top_dist_squared) {
        bottom_to_top = center - bottom;
      } else {  // bottom_dist_squared >= top_dist_squared
        bottom_to_top = top - center;
      }

      Eigen::Vector3f normal = left_to_right.cross(bottom_to_top);
      const float length = normal.norm();
      if (!(length > 1e-6f)) {
        normal = Eigen::Vector3f(0, 0, -1);
      } else {
        normal.normalize();
      }

      const Eigen::Vector3f xvec =
          ((center + left) * 0.5) - ((center + right) * 0.5);
      const Eigen::Vector3f yvec =
          (center + top) * 0.5 - (center + bottom) * 0.5;

      // const Eigen::Vector3f normal = xvec.cross(yvec).normalized();
      n_acc[row][col][0] = normal[0];
      n_acc[row][col][1] = normal[1];
      n_acc[row][col][2] = normal[2];
    }
  }
}

void ComputeCentralDifferences(const torch::Tensor xyz_image,
                               const torch::Tensor mask_image,
                               torch::Tensor out_normals) {
  if (xyz_image.is_cuda()) {
    ComputeCentralDifferences_gpu(xyz_image, mask_image, out_normals);
  } else {
    return AT_DISPATCH_ALL_TYPES(
        xyz_image.scalar_type(), "CompNormalsCentralDiff_cpu_kernel", ([&] {
          ComputeCentralDifferences_cpu_kernel<scalar_t>(xyz_image, mask_image,
                                                         out_normals);
        }));
  }
}
}  // namespace

void EstimateNormals(const torch::Tensor xyz_image,
                     const torch::Tensor mask_image, torch::Tensor out_normals,
                     EstimateNormalsMethod method) {
  switch (method) {
    case kCentralDifferences:
      ComputeCentralDifferences(xyz_image, mask_image, out_normals);
      break;
    case kAverage8:
      ComputeAverage8(xyz_image.cpu(), mask_image.cpu(), out_normals);
      break;
  };
}
}  // namespace fiontb
