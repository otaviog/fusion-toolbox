#include "elastic_fusion.hpp"

#include "camera.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "surfel_fusion_common.hpp"

namespace fiontb {
namespace {

template <Device dev>
struct FindMergeKernel {
  const IndexMapAccessor<dev> model_indexmap;
  const IndexMapAccessor<dev> live_indexmap;

  int scale, search_size;
  const int time;

  MergeMap<dev> merge_map;
  typename Accessor<dev, int64_t, 2>::T new_surfel_map;

  FindMergeKernel(const IndexMap &model_indexmap, const IndexMap &live_indexmap,
                  int search_size_, int time, torch::Tensor merge_map,
                  torch::Tensor new_surfel_map)
      : model_indexmap(model_indexmap),
        live_indexmap(live_indexmap),
        time(time),
        merge_map(merge_map),
        new_surfel_map(Accessor<dev, int64_t, 2>::Get(new_surfel_map)) {
    float scale_ =
        float(model_indexmap.get_height()) / float(live_indexmap.get_height());
    scale = int(scale_);
    search_size = int(scale * search_size_);
  }

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int row, int col) {
    new_surfel_map[row][col] = -1;

    if (live_indexmap.empty(row, col)) return;

    const int neighborhood[4][2] = {{-1, 0}, {0, 1}, {0, -1}, {0, 1}};
    for (int k = 0; k < 4; ++k) {
      const int r = row + neighborhood[k][0];
      const int c = col + neighborhood[k][1];
      if (r >= 0 && r < live_indexmap.height() && c >= 0 &&
          c < live_indexmap.width()) {
        if (live_indexmap.empty(r, c)) return;
      }
    }

    const int live_index = live_indexmap.to_linear_index(row, col);

    const Vector<float, 3> live_pos(live_indexmap.position(row, col));

    if (row % 2 == time % 2 || col % 2 == time % 2) return;

    const Vector<float, 3> ray(live_pos[0] / live_pos[2],
                               live_pos[1] / live_pos[2], 1);
    float lambda = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1);

    const Vector<float, 3> live_normal(live_indexmap.normal(row, col));

    const int xstart = max(col * scale - search_size, 0);
    const int xend =
        min(col * scale + search_size, int(model_indexmap.width()) - 1);

    const int ystart = max(row * scale - search_size, 0);
    const int yend =
        min(row * scale + search_size, int(model_indexmap.height()) - 1);

    float best_dist = NumericLimits<dev, float>::infinity();
    int best_row, best_col;
    best_row = best_col = -1;

    for (int krow = ystart; krow <= yend; krow++) {
      for (int kcol = xstart; kcol <= xend; kcol++) {
        if (model_indexmap.empty(krow, kcol)) continue;

        const Vector<float, 3> model_pos = model_indexmap.position(krow, kcol);
        if (abs((model_pos[2] * lambda) - (live_pos[2] * lambda)) >= 0.05)
          continue;

        const float dist = ray.cross(model_pos).norm() / ray.norm();

        const Vector<float, 3> normal = model_indexmap.normal(krow, kcol);
        if (dist < best_dist && (GetVectorsAngle(normal, live_normal) < .5 ||
                                 abs(normal[2]) < 0.75f)) {
          best_dist = dist;
          best_row = krow;
          best_col = kcol;
        }
      }
    }

    if (best_dist < NumericLimits<dev, float>::infinity()) {
      merge_map.Set(best_row, best_col, int32_t(double(best_dist) * 1e15),
                    live_index);
    } else {
      new_surfel_map[row][col] = live_indexmap.index(row, col);
    }
  }
};

template <Device dev>
struct UpdateKernel {
  const IndexMapAccessor<dev> model_indexmap;
  const IndexMapAccessor<dev> live_indexmap;

  const typename Accessor<dev, float, 3>::T live_features;
  typename Accessor<dev, int32_t, 3>::T model_merge_map;

  const RTCamera<float> rt_cam;
  const Eigen::Matrix3f normal_transform_matrix;
  const int time;

  SurfelModelAccessor<dev> model;

  UpdateKernel(const IndexMap &model_indexmap, const IndexMap &live_indexmap,
               const torch::Tensor &live_features,
               const torch::Tensor &model_merge_map,
               const torch::Tensor &rt_cam, int time, MappedSurfelModel model)
      : model_indexmap(model_indexmap),
        live_indexmap(live_indexmap),
        live_features(Accessor<dev, float, 3>::Get(live_features)),
        model_merge_map(Accessor<dev, int32_t, 3>::Get(model_merge_map)),
        rt_cam(rt_cam),
        time(time),
        model(model) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    if (model_indexmap.empty(row, col)) return;
    const int32_t live_index = model_merge_map[row][col][1];
    if (live_index == INT_MAX) return;
    int live_row, live_col;
    live_indexmap.to_rowcol_index(live_index, &live_row, &live_col);

    const int32_t model_target = model_indexmap.index(row, col);

    const float live_conf = live_indexmap.confidence(live_row, live_col);
    const float model_conf = model.confidences[model_target];
    const float conf_total = live_conf + model_conf;

    const float live_radius = live_indexmap.radius(live_row, live_col);
    const float model_radius = model.radii[model_target];

    if (live_radius < (1.0 + 0.5) * model_radius) {
      const Vector<float, 3> live_pos(
          live_indexmap.position(live_row, live_col));
      const Vector<float, 3> live_world_pos = rt_cam.Transform(live_pos);
      model.set_position(model_target,
                         (model.position(model_target) * model_conf +
                          live_world_pos * live_conf) /
                             conf_total);
      const Vector<float, 3> live_normal(
          live_indexmap.normal(live_row, live_col));
      const Vector<float, 3> live_world_normal =
          rt_cam.TransformNormal(live_normal);
      model.set_normal(model_target, (model.normal(model_target) * model_conf +
                                      live_world_normal * live_conf) /
                                         conf_total);

      const Vector<float, 3> live_color(
          live_indexmap.color(live_row, live_col));
      model.set_color(model_target, (model.color(model_target) * model_conf +
                                     live_color * live_conf) /
                                        conf_total);
      const int64_t feature_size =
          min(model.features.size(0), live_features.size(0));
      const int live_height = live_features.size(1);
      for (int64_t i = 0; i < feature_size; ++i) {
        const float model_feat_channel = model.features[i][model_target];
        // Indexmap comes up side down
        const float live_feat_channel =
            live_features[i][live_height - 1 - live_row][live_col];

        model.features[i][model_target] =
            (model_feat_channel * model_conf + live_feat_channel * live_conf) /
            conf_total;
      }
    }
    model.confidences[model_target] = conf_total;
    model.times[model_target] = time;
  }
};

}  // namespace

void ElasticFusionOp::Update(const IndexMap &model_indexmap,
                             const IndexMap &live_indexmap,
                             const torch::Tensor &live_features,
                             MappedSurfelModel model,
                             const torch::Tensor &rt_cam, int search_size,
                             int time, torch::Tensor merge_map,
                             torch::Tensor new_surfels_map) {
  auto reference_dev = model_indexmap.get_device();

  live_indexmap.CheckDevice(reference_dev);
  FTB_CHECK_DEVICE(reference_dev, live_features);
  model_indexmap.CheckDevice(reference_dev);
  model.CheckDevice(reference_dev);

  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, merge_map);
  FTB_CHECK_DEVICE(reference_dev, new_surfels_map);

  if (reference_dev.is_cuda()) {
    FindMergeKernel<kCUDA> find_kernel(model_indexmap, live_indexmap,
                                       search_size, time, merge_map,
                                       new_surfels_map);
    Launch2DKernelCUDA(find_kernel, live_indexmap.get_width(),
                       live_indexmap.get_height());

    UpdateKernel<kCUDA> merge_kernel(model_indexmap, live_indexmap,
                                     live_features, merge_map, rt_cam, time,
                                     model);
    Launch2DKernelCUDA(merge_kernel, model_indexmap.get_width(),
                       model_indexmap.get_height());
  } else {
    FindMergeKernel<kCPU> find_kernel(model_indexmap, live_indexmap,
                                      search_size, time, merge_map,
                                      new_surfels_map);
    Launch2DKernelCPU(find_kernel, live_indexmap.get_width(),
                      live_indexmap.get_height());
    UpdateKernel<kCPU> merge_kernel(model_indexmap, live_indexmap,
                                    live_features, merge_map, rt_cam, time,
                                    model);
    Launch2DKernelCPU(merge_kernel, model_indexmap.get_width(),
                      model_indexmap.get_height());
  }
}
}  // namespace fiontb
