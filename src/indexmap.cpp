#include "indexmap.hpp"

#include "eigen_common.hpp"

using namespace std;

namespace fiontb {
const int MAX_SLOTS = 9;

IndexMap::IndexMap(torch::Tensor proj_points, int width, int height) {
  grid_ = torch::full({height, width, MAX_SLOTS}, -1, torch::kInt64);
  model_points_ = proj_points;

  const auto pp_acc = proj_points.accessor<float, 2>();
  auto grid_acc = grid_.accessor<long, 3>();

  for (long row = 0; row < proj_points.size(0); ++row) {
    const int x = pp_acc[row][0];
    const int y = pp_acc[row][1];
    int k = 0;

    for (; k < MAX_SLOTS - 1; ++k) {
      if (grid_acc[y][x][k] == -1) break;
    }

    grid_acc[y][x][k] = row;
  }
}

pair<torch::Tensor, torch::Tensor> IndexMap::Query(
    const torch::Tensor &query_points, int query_k) {
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

  for (long row = 0; row < qp_acc.size(0); ++row) {
    const float x = qp_acc[row][0];
    const float y = qp_acc[row][1];
    const Eigen::Vector3f qpoint(x, y, qp_acc[row][2]);

    std::vector<std::pair<long, float>> distances;

    for (int wnd_y = -1; wnd_y < 2; ++wnd_y) {
      for (int wnd_x = -1; wnd_x < 2; ++wnd_x) {
        const int cy = y - wnd_y;
        if (cy < 0 || cy >= grid_.size(0)) continue;

        const int cx = x - wnd_x;
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
