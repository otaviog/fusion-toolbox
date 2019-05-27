#include "surfel_fusion.hpp"

#include "eigen_common.hpp"

using namespace std;

namespace fiontb {

torch::Tensor FilterSearch(torch::Tensor dist_mtx, torch::Tensor idx_mtx,
                           torch::Tensor live_normals,
                           torch::Tensor model_normals, float min_dist,
                           float min_normal_dot) {
  auto dist_acc = dist_mtx.accessor<float, 2>();
  auto idx_acc = idx_mtx.accessor<int64_t, 2>();
  auto ln_acc = live_normals.accessor<float, 2>();
  auto mn_acc = model_normals.accessor<float, 2>();

  torch::Tensor choosen = torch::full(
      {dist_mtx.size(0)}, -1, torch::TensorOptions().dtype(torch::kInt64));
  auto chs_acc = choosen.accessor<int64_t, 1>();

  for (int row = 0; row < dist_mtx.size(0); ++row) {
    const Eigen::Vector3f live_normal(ln_acc[row][0], ln_acc[row][1],
                                      ln_acc[row][2]);
    for (int col = 0; col < dist_mtx.size(1); ++col) {
      if (dist_acc[row][col] > min_dist) {
        break;
      }

      const int which_model_normal = idx_acc[row][col];
      const Eigen::Vector3f model_normal(mn_acc[which_model_normal][0],
                                         mn_acc[which_model_normal][1],
                                         mn_acc[which_model_normal][2]);
      if (live_normal.dot(model_normal) < min_normal_dot) {
        continue;
      }

      chs_acc[row] = which_model_normal;
      break;
    }
  }

  return choosen;
}

}  // namespace fiontb
