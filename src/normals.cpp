#include "normals.hpp"

#include <iostream>

#include "eigen_common.hpp"

namespace fiontb {
inline Eigen::Vector3f Normal(const Eigen::Vector3f &p0,
                              const Eigen::Vector3f &p1,
                              const Eigen::Vector3f &p2) {
  return (p1 - p0).cross(p2 - p0).normalized();
}

torch::Tensor CalculateFrameNormals1(const torch::Tensor xyz_image,
                                    const torch::Tensor mask_image) {
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

        norm += Normal(center,
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

  return normal_image;
}

torch::Tensor CalculateFrameNormals(const torch::Tensor xyz_image,
                                    const torch::Tensor mask_image) {
  torch::Tensor normal_image =
      torch::empty({xyz_image.size(0), xyz_image.size(1), 3}, torch::kFloat);
  auto n_acc = normal_image.accessor<float, 3>();

  const auto xyz_a = xyz_image.accessor<float, 3>();
  const auto mask_a = mask_image.accessor<uint8_t, 2>();

  const int iwidth = xyz_image.size(1);
  const int iheight = xyz_image.size(0);

  for (int row = 1; row < xyz_image.size(0) - 1; ++row) {
    for (int col = 1; col < xyz_image.size(1) - 1; ++col) {
      n_acc[row][col][0] = n_acc[row][col][1] = n_acc[row][col][2] = 0.0f;

      if (mask_a[row][col] == 0) continue;

      Eigen::Vector3f center(xyz_a[row][col][0], xyz_a[row][col][1],
                             xyz_a[row][col][2]);
      
      const Eigen::Vector3f left(xyz_a[row][col - 1][0],
                                 xyz_a[row][col - 1][1],
                                 xyz_a[row][col - 1][2]);
      const Eigen::Vector3f right(xyz_a[row][col + 1][0],
                                  xyz_a[row][col + 1][1],
                                  xyz_a[row][col + 1][2]);

      const Eigen::Vector3f top(xyz_a[row - 1][col][0],
                                xyz_a[row - 1][col][1],
                                xyz_a[row - 1][col][2]);
      const Eigen::Vector3f bottom(xyz_a[row + 1][col][0],
                                   xyz_a[row + 1][col][1],
                                   xyz_a[row + 1][col][2]);

      const Eigen::Vector3f xvec =
          ((center + left) * 0.5) - ((center + right) * 0.5);
      const Eigen::Vector3f yvec =
          (center + top) * 0.5 - (center + bottom) * 0.5;

      const Eigen::Vector3f normal = xvec.cross(yvec).normalized();
      n_acc[row][col][0] = normal[0];
      n_acc[row][col][1] = normal[1];
      n_acc[row][col][2] = normal[2];
    }
  }

  return normal_image;
}
}  // namespace fiontb
