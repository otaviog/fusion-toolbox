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
  const SurfelCloudAccessor<dev> live_surfels;

  const KCamera<dev, float> kcam;

  int scale, search_size;
  const int time;

  MergeMap<dev> merge_map;
  typename Accessor<dev, bool, 1>::T new_surfel_map;

  FindMergeKernel(const IndexMap &model_indexmap,
                  const SurfelCloud &live_surfels, const torch::Tensor &kcam,
                  int scale, int search_size_, int time,
                  torch::Tensor merge_map, torch::Tensor new_surfel_map)
      : model_indexmap(model_indexmap),
        live_surfels(live_surfels),
        kcam(kcam),
        scale(scale),
        time(time),
        merge_map(merge_map),
        new_surfel_map(Accessor<dev, bool, 1>::Get(new_surfel_map)) {
    search_size = int(scale * search_size_);
  }

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int live_index) {
    new_surfel_map[live_index] = false;

    const Eigen::Vector3f live_pos(live_surfels.position(live_index));
    int col, row;
    kcam.Projecti(live_pos, col, row);

    if (row % 2 == time % 2 || col % 2 == time % 2) return;

    new_surfel_map[live_index] = true;

    const Vector<float, 3> ray(live_pos[0] / live_pos[2],
                               live_pos[1] / live_pos[2], 1);
    const float lambda = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1);

    const Vector<float, 3> live_normal(live_surfels.normal(live_index));

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
      new_surfel_map[live_index] = false;
    }
  }
};

template <Device dev>
struct UpdateKernel {
  const IndexMapAccessor<dev> model_indexmap;
  const SurfelCloudAccessor<dev> live_surfels;

  typename Accessor<dev, int32_t, 3>::T model_merge_map;

  const RTCamera<float> rt_cam;
  const int time;

  SurfelModelAccessor<dev> model;

  UpdateKernel(const IndexMap &model_indexmap, const SurfelCloud &live_surfels,
               const torch::Tensor &model_merge_map,
               const torch::Tensor &rt_cam, int time, MappedSurfelModel model)
      : model_indexmap(model_indexmap),
        live_surfels(live_surfels),
        model_merge_map(Accessor<dev, int32_t, 3>::Get(model_merge_map)),
        rt_cam(rt_cam),
        time(time),
        model(model) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    if (model_indexmap.empty(row, col)) return;
    const int32_t live_index = model_merge_map[row][col][1];

    if (live_index == INT_MAX) return;

    const int32_t model_target = model_indexmap.index(row, col);

    const float live_conf = live_surfels.confidences[live_index];
    const float model_conf = model.confidences[model_target];
    const float conf_total = live_conf + model_conf;

    const float live_radius = live_surfels.radii[live_index];
    const float model_radius = model.radii[model_target];

    if (live_radius < (1.0 + 0.5) * model_radius) {
      const Vector<float, 3> live_pos(live_surfels.position(live_index));
      const Vector<float, 3> live_world_pos = rt_cam.Transform(live_pos);
      model.set_position(model_target,
                         (model.position(model_target) * model_conf +
                          live_world_pos * live_conf) /
                             conf_total);
      const Vector<float, 3> live_normal(live_surfels.normal(live_index));
      const Vector<float, 3> live_world_normal =
          rt_cam.TransformNormal(live_normal);
      model.set_normal(model_target, (model.normal(model_target) * model_conf +
                                      live_world_normal * live_conf) /
                                         conf_total);

      const Vector<float, 3> live_color(live_surfels.color(live_index));
      model.set_color(model_target, (model.color(model_target) * model_conf +
                                     live_color * live_conf) /
                                        conf_total);
      const int64_t feature_size =
          min(model.features.size(0), live_surfels.features.size(0));
      const int live_height = live_surfels.features.size(1);
      for (int64_t i = 0; i < feature_size; ++i) {
        const float model_feat_channel = model.features[i][model_target];
        const float live_feat_channel = live_surfels.features[i][live_index];

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
                             const SurfelCloud &live_surfels,
                             MappedSurfelModel model, const torch::Tensor &kcam,
                             const torch::Tensor &rt_cam, int search_size,
                             int time, int scale, torch::Tensor merge_map,
                             torch::Tensor new_surfels_map) {
  auto reference_dev = model_indexmap.get_device();

  live_surfels.CheckDevice(reference_dev);
  model_indexmap.CheckDevice(reference_dev);
  model.CheckDevice(reference_dev);

  FTB_CHECK_DEVICE(reference_dev, merge_map);
  FTB_CHECK_DEVICE(reference_dev, new_surfels_map);

  if (reference_dev.is_cuda()) {
    FindMergeKernel<kCUDA> find_kernel(model_indexmap, live_surfels, kcam,
                                       scale, search_size, time, merge_map,
                                       new_surfels_map);
    Launch1DKernelCUDA(find_kernel, live_surfels.get_size());

    UpdateKernel<kCUDA> update_kernel(model_indexmap, live_surfels, merge_map,
                                      rt_cam, time, model);
    Launch2DKernelCUDA(update_kernel, model_indexmap.get_width(),
                       model_indexmap.get_height());
  } else {
    FindMergeKernel<kCPU> find_kernel(model_indexmap, live_surfels, kcam, scale,
                                      search_size, time, merge_map,
                                      new_surfels_map);

    Launch1DKernelCPU(find_kernel, live_surfels.get_size());
    UpdateKernel<kCPU> update_kernel(model_indexmap, live_surfels, merge_map,
                                     rt_cam, time, model);

    Launch2DKernelCPU(update_kernel, model_indexmap.get_width(),
                      model_indexmap.get_height());
  }
}
}  // namespace fiontb
