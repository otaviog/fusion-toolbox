#include "normals.hpp"

#include "eigen_common.hpp"

namespace fiontb {
inline Eigen::Vector3f Normal(const Eigen::Vector3f &p0,
                              const Eigen::Vector3f &p1,
                              const Eigen::Vector3f &p2) {
  return (p1 - p0).cross(p2 - p0).normalized();
}

torch::Tensor CalculateFrameNormals(const torch::Tensor xyz_image,
                                    const torch::Tensor mask_image) {
  torch::Tensor normal_image =
      torch::empty({xyz_image.size(0), xyz_image.size(1), 3}, torch::kFloat);
  auto n_acc = normal_image.accessor<float, 3>();

  const auto xyz_a = xyz_image.accessor<float, 3>();
  const auto mask_a = mask_image.accessor<uint8_t, 2>();

  const int where[][2] = {{0, 1}, {1, 1}, {1, -1}, {0, -1}, {-1, -1}, {1, 0}};

  for (int row = 0; row < xyz_image.size(0); ++row) {
    for (int col = 0; col < xyz_image.size(1); ++col) {
      Eigen::Vector3f norm(0, 0, 0);
      int count = 0;
      for (int i = 0; i < 6; i++) {
        const int r1 = row + where[i][0];
        if (r1 < 0 || r1 >= xyz_image.size(0)) continue;

        const int c1 = col + where[i][1];
        if (c1 < 0 || c1 >= xyz_image.size(1)) continue;

        if (mask_a[r1][c1] == 0) continue;

        const int j = (i + 1) % 6;
        const int r2 = row + where[j][0];
        if (r2 < 0 || r2 >= xyz_image.size(0)) continue;

        const int c2 = col + where[j][1];
        if (c2 < 0 || c2 >= xyz_image.size(1)) continue;

        if (mask_a[r2][c2] == 0) continue;
        norm += Normal(Eigen::Vector3f(xyz_a[row][col][0], xyz_a[row][col][1],
                                       xyz_a[row][col][2]),
                       Eigen::Vector3f(xyz_a[r1][c1][0], xyz_a[r1][c1][1],
                                       xyz_a[r1][c1][2]),
                       Eigen::Vector3f(xyz_a[r2][c2][0], xyz_a[r2][c2][1],
                                       xyz_a[r2][c2][2]));
        ++count;
      }
      if (count > 0) {
        norm = norm / count;
        norm.normalize();
        n_acc[row][col][0] = norm[0];
        n_acc[row][col][1] = norm[1];
        n_acc[row][col][2] = norm[2];
      } else {
        n_acc[row][col][0] = n_acc[row][col][1] = n_acc[row][col][2] = 0.0f;
      }
    }
  }

  return normal_image;
}
}  // namespace fiontb
