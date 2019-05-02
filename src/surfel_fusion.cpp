#include "surfel_fusion.hpp"
namespace fiontb {

torch::Tensor FilterSearch(torch::Tensor dist_mtx, torch::Tensor idx_mtx,
                           torch::Tensor live_normals,
                           torch::Tensor model_normals, float min_dist,
                           float min_normal_dot) {
  auto dist_acc = dist_mtx.accessor<float, 2>();
  auto idx_acc = idx_mtx.accessor<uint64_t, 2>();
  auto ln_acc = live_normals.accessor<float, 2>();
  auto mn_acc = model_normals.accessor<float, 2>();

  torch::Tensor choosen = torch::full(
      {dist_mtx.size(0)}, -1, torch::TensorOptions().dtype(torch::kUInt8));
  auto chs_acc = choosen.accessor<uint8_t, 1>();

  for (int row = 0; row < dist_mtx.size(0); ++row) {
    for (int col = 0; col < dist_mtx.size(1); ++col) {
      if (dist_acc[row][col] > min_dist) {
        break;
      }

      const int which_model_normal = idx_acc[row][col];
      if (torch::dot(live_normals[row], model_normals[which_model_normal])
              .item()
              .to<float>() < min_normal_dot) {
        continue;
      }

      chs_acc[row] = which_model_normal;
      break;
    }
  }

  return choosen;
}
}  // namespace fiontb
