#include "surfel_fusion.hpp"

#include <mutex>

#include "camera.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "merge_map.hpp"
#include "surfel_fusion_common.hpp"

namespace slamtb {
namespace {
template <Device dev>
struct FindMergeKernel {
  const IndexMapAccessor<dev> indexmap;
  const SurfelCloudAccessor<dev> live_surfels;
  const KCamera<dev, float> kcam;

  int scale, search_size;
  const float max_normal_angle;
  const int time;

  MergeMap<dev> merge_map;
  typename Accessor<dev, bool, 1>::T new_surfels_map;

  FindMergeKernel(const IndexMap &indexmap,
                  const SurfelCloud &live_surfels, const torch::Tensor &kcam,
                  float max_normal_angle, int search_size_, int time, int scale,
                  torch::Tensor merge_map, torch::Tensor new_surfels_map)
      : indexmap(indexmap),
        live_surfels(live_surfels),
        kcam(kcam),
        scale(scale),
        max_normal_angle(max_normal_angle),
        time(time),
        merge_map(merge_map),
        new_surfels_map(Accessor<dev, bool, 1>::Get(new_surfels_map)) {
    search_size = int(scale * search_size_);
  }

#pragma nv_exec_check_disable
  STB_DEVICE_HOST void operator()(int live_index) {
    new_surfels_map[live_index] = true;

    const Eigen::Vector3f live_pos(live_surfels.point(live_index));
    int u, v;
    kcam.Projecti(live_pos, u, v);

    // Undo the z division done in backprojection
    const Vector<float, 3> ray(live_pos[0] / live_pos[2],
                               live_pos[1] / live_pos[2], 1);
    const float lambda = ray.norm();
    const Vector<float, 3> live_normal(live_surfels.normal(live_index));

    const int xstart = max(u * scale - search_size, 0);
    const int xend =
        min(u * scale + search_size, int(indexmap.width()) - 1);

    const int ystart = max(v * scale - search_size, 0);
    const int yend =
        min(v * scale + search_size, int(indexmap.height()) - 1);

    float best_dist = NumericLimits<dev, float>::infinity();
    int best_row = -1, best_col = -1;

    for (int krow = ystart; krow <= yend; krow++) {
      for (int kcol = xstart; kcol <= xend; kcol++) {
        if (indexmap.empty(krow, kcol)) continue;

        const Vector<float, 3> model_pos = indexmap.point(krow, kcol);
        if (abs((model_pos[2] * lambda) - (live_pos[2] * lambda)) >= 0.05)
          continue;
        const float ray_dist = model_pos.cross(ray).norm() / ray.norm();

        const Vector<float, 3> model_normal = indexmap.normal(krow, kcol);
        if (ray_dist < best_dist &&
            (GetVectorsAngle(model_normal, live_normal) < max_normal_angle ||
             abs(model_normal[2]) < 0.75f)) {
          best_dist = ray_dist;
          best_row = krow;
          best_col = kcol;
        }
      }
    }

    if (best_dist < NumericLimits<dev, float>::infinity()) {
      merge_map.Set(best_row, best_col, best_dist, live_index);
      new_surfels_map[live_index] = false;
    }
  }
};

template <Device dev>
struct SelectUpdatableIndexKernel {
  const IndexMapAccessor<dev> indexmap;
  MergeMapAccessor<dev> model_merge_map;
  typename Accessor<dev, int64_t, 2>::T merge_corresp;

  SelectUpdatableIndexKernel(const IndexMap &indexmap,
                             const torch::Tensor &model_merge_map,
                             torch::Tensor &merge_corresp)
      : indexmap(indexmap),
        model_merge_map(model_merge_map),
        merge_corresp(Accessor<dev, int64_t, 2>::Get(merge_corresp)) {}

  STB_DEVICE_HOST void operator()(int row, int col) {
    if (indexmap.empty(row, col) || model_merge_map.empty(row, col))
      return;
    const int32_t model_target = indexmap.index(row, col);
    const int32_t live_index = model_merge_map(row, col);

    const int index = indexmap.to_linear_index(row, col);
    merge_corresp[index][0] = model_target;
    merge_corresp[index][1] = live_index;
  }
};

template <Device dev>
struct UpdateKernel {
  const typename Accessor<dev, int64_t, 2>::T merge_corresp;
  const SurfelCloudAccessor<dev> live_surfels;
  const RigidTransform<float> rt_cam;
  const int time;
  SurfelModelAccessor<dev> model;

  UpdateKernel(const torch::Tensor &merge_corresp,
               const SurfelCloud &live_surfels, const torch::Tensor &rt_cam,
               int time, MappedSurfelModel model)
      : merge_corresp(Accessor<dev, int64_t, 2>::Get(merge_corresp)),
        live_surfels(live_surfels),
        rt_cam(rt_cam),
        time(time),
        model(model) {}

  STB_DEVICE_HOST void operator()(int corresp) {
    const int32_t model_target = merge_corresp[corresp][0];
    const int32_t live_index = merge_corresp[corresp][1];

    const float live_conf = live_surfels.confidences[live_index];
    const float model_conf = model.confidences[model_target];
    const float conf_total = live_conf + model_conf;

    model.confidences[model_target] = conf_total;
    model.times[model_target] = time;

    const float live_radius = live_surfels.radii[live_index];
    const float model_radius = model.radii[model_target];

    if (live_radius >= (1.0 + 0.5) * model_radius) {
      return;
    }

    const Vector<float, 3> live_pos(live_surfels.point(live_index));
    const Vector<float, 3> live_world_pos = rt_cam.Transform(live_pos);
    model.set_point(model_target, (model.point(model_target) * model_conf +
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
};
}  // namespace

void SurfelFusionOp::FindUpdatable(const IndexMap &indexmap,
                                   const SurfelCloud &live_surfels,
                                   const torch::Tensor &kcam,
                                   float max_normal_angle, int search_size,
                                   int time, int scale, torch::Tensor merge_map,
                                   torch::Tensor new_surfels_map,
                                   torch::Tensor merge_corresp) {
  auto reference_dev = indexmap.get_device();
  live_surfels.CheckDevice(reference_dev);
  STB_CHECK_DEVICE(reference_dev, merge_map);
  STB_CHECK_DEVICE(reference_dev, new_surfels_map);

  if (reference_dev.is_cuda()) {
    FindMergeKernel<kCUDA> find_kernel(indexmap, live_surfels, kcam,
                                       max_normal_angle, search_size, time,
                                       scale, merge_map, new_surfels_map);
    Launch1DKernelCUDA(find_kernel, live_surfels.get_size());
    SelectUpdatableIndexKernel<kCUDA> select_kernel(indexmap, merge_map,
                                                    merge_corresp);
    Launch2DKernelCUDA(select_kernel, indexmap.get_width(),
                       indexmap.get_height());

  } else {
    FindMergeKernel<kCPU> find_kernel(indexmap, live_surfels, kcam,
                                      max_normal_angle, search_size, time,
                                      scale, merge_map, new_surfels_map);

    Launch1DKernelCPU(find_kernel, live_surfels.get_size());
    SelectUpdatableIndexKernel<kCPU> select_kernel(indexmap, merge_map,
                                                   merge_corresp);
    Launch2DKernelCPU(select_kernel, indexmap.get_width(),
                      indexmap.get_height());
  }
}

void SurfelFusionOp::Update(const torch::Tensor &merge_corresp,
                            const SurfelCloud &live_surfels,
                            MappedSurfelModel model,
                            const torch::Tensor &rt_cam, int time) {
  auto reference_dev = merge_corresp.device();

  live_surfels.CheckDevice(reference_dev);
  model.CheckDevice(reference_dev);

  if (reference_dev.is_cuda()) {
    UpdateKernel<kCUDA> update_kernel(merge_corresp, live_surfels, rt_cam, time,
                                      model);

    Launch1DKernelCUDA(update_kernel, merge_corresp.size(0));
  } else {
    UpdateKernel<kCPU> update_kernel(merge_corresp, live_surfels, rt_cam, time,
                                     model);

    Launch1DKernelCPU(update_kernel, merge_corresp.size(0));
  }
}
}  // namespace slamtb
