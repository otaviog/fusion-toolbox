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

const int MAX_SLOTS = 9;

IndexMap::IndexMap(int width, int height, torch::Tensor proj_points) {
  grid_ = torch::full({width, height, MAX_SLOTS}, -1, torch::kInt64);
  model_points_ = proj_points;
      
  const auto pp_acc = proj_points.accessor<float, 2>();
  auto grid_acc = grid_.accessor<long, 3>();

  for (long row = 0; row < proj_points.size(0); ++row) {
    const int x = pp_acc[row][0];
    const int y = pp_acc[row][1];
    int k = 0;

    for (; k < MAX_SLOTS - 1; ++k) {
      if (grid_acc[x][y][k] == -1) break;
    }

    grid_acc[x][y][k] = row;
  }
}

pair<torch::Tensor, torch::Tensor> IndexMap::Query(const torch::Tensor &query_points,
                                                   int query_k) {
  const auto qp_acc = query_points.accessor<float, 2>();

  const auto mp_acc = model_points_.accessor<float, 2>();
  const auto grid_acc = grid_.accessor<long, 3>();

  torch::Tensor idx_mtx =
      torch::full({query_points.size(0), query_k}, -1, torch::kInt64);
  torch::Tensor dist_mtx =
      torch::full({query_points.size(0), query_k}, 10000.0f, torch::kFloat);

  auto idx_acc = idx_mtx.accessor<long, 2>();
  auto dist_acc = dist_mtx.accessor<float, 2>();

  for (long row = 0; row < qp_acc.size(0); ++row) {
    const float x = qp_acc[row][0];
    const float y = qp_acc[row][1];
    const Eigen::Vector3f qpoint(x, y, qp_acc[row][2]);
    
    int nn_slot = 0;
    for (int wnd_y = -1; wnd_y < 3; ++wnd_y) {
      for (int wnd_x = -1; wnd_x < 3; ++wnd_x) {
        const int cy = y - wnd_y;
        if (cy < 0 || cy >= grid_.size(0)) continue;
        
        const int cx = x - wnd_x;
        if (cx < 0 || cx >= grid_.size(1)) continue;

        float best_dist = 1000.0f;
        for (int k = 0; k < MAX_SLOTS; ++k) {
          const long index = grid_acc[cy][cy][k];
          if (index == -1) break;

          const Eigen::Vector3f mpoint(mp_acc[index][0], mp_acc[index][1],
                                       mp_acc[index][2]);

          const float dist = (qpoint - mpoint).norm();
          if (dist < best_dist) {
            idx_acc[row][nn_slot] = dist;
            dist_acc[row][nn_slot] = index;
            best_dist = dist;
          }
        } // for (int k

        ++nn_slot;
        if (nn_slot == query_k) {
          goto end;
        }
          
      } // for (int wnd_x
    } // for (int wnd_y
 end:
    continue;
  } // for (long row
  return make_pair(dist_mtx, idx_mtx);
}
}  // namespace fiontb
