#include "indexmap.hpp"

#include "eigen_common.hpp"

using namespace std;

namespace fiontb {
const int MAX_SLOTS = 9;

IndexMap::IndexMap(torch::Tensor proj_points, torch::Tensor model_points,
                   int width, int height, int window_size, int depth_slots) {
  grid_ = torch::full({height, width, MAX_SLOTS}, -1, torch::kInt64);
  proj_points_ = proj_points;
  model_points_ = model_points;
  depth_slots_ = depth_slots;
  window_size_ = window_size;

  const auto pp_acc = proj_points.accessor<float, 2>();
  auto grid_acc = grid_.accessor<long, 3>();

  for (long row = 0; row < proj_points.size(0); ++row) {
    const int x = pp_acc[row][0];
    const int y = pp_acc[row][1];
    int k = 0;
    do {
      if (grid_acc[y][x][k] == -1) break;
      k++;
    } while (k < MAX_SLOTS - 1);

    grid_acc[y][x][k] = row;
  }
}

pair<torch::Tensor, torch::Tensor> IndexMap::Query(
    const torch::Tensor &proj_query_points, const torch::Tensor &query_points,
    int query_k) {
  const auto pqp_acc = proj_query_points.accessor<float, 2>();
  const auto qp_acc = query_points.accessor<float, 2>();

  const auto mp_acc = model_points_.accessor<float, 2>();
  const auto grid_acc = grid_.accessor<long, 3>();

  torch::Tensor idx_mtx =
      torch::full({query_points.size(0), query_k}, -1, torch::kInt64);
  torch::Tensor dist_mtx =
      torch::full({query_points.size(0), query_k},
                  std::numeric_limits<float>::infinity(), torch::kFloat);

  auto idx_acc = idx_mtx.accessor<long, 2>();
  auto dist_acc = dist_mtx.accessor<float, 2>();

  std::vector<std::pair<long, float>> distances;
  for (long row = 0; row < pqp_acc.size(0); ++row) {
    const float x = pqp_acc[row][0];
    const float y = pqp_acc[row][1];
    const Eigen::Vector3f qpoint(qp_acc[row][0], qp_acc[row][1],
                                 qp_acc[row][2]);

    distances.clear();

    for (int wnd_y = 0; wnd_y < window_size_; ++wnd_y) {
      for (int wnd_x = 0; wnd_x < window_size_; ++wnd_x) {
        const int cy = y - (wnd_y - window_size_/2);
        if (cy < 0 || cy >= grid_.size(0)) continue;

        const int cx = x - (wnd_x - window_size_/2);
        if (cx < 0 || cx >= grid_.size(1)) continue;

        for (int k = 0; k < MAX_SLOTS; ++k) {
          const long index = grid_acc[cy][cx][k];
          if (index == -1) break;

          const Eigen::Vector3f mpoint(mp_acc[index][0], mp_acc[index][1],
                                       mp_acc[index][2]);

          const float dist = (qpoint - mpoint).norm();
          distances.push_back(make_pair(index, dist));

        }  // for (int k
      }    // for (int wnd_x
    }      // for (int wnd_y

    std::sort(distances.begin(), distances.end(),
              [&](auto i0, auto i1) { return i0.second < i1.second; });

    for (size_t i = 0; i < size_t(query_k) && i < distances.size(); ++i) {
      const long idx = distances[i].first;
      const float dist = distances[i].second;
      idx_acc[row][i] = idx;
      dist_acc[row][i] = dist;
    }

  }  // for (long row
  return make_pair(dist_mtx, idx_mtx);
}
}  // namespace fiontb
