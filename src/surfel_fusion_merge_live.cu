#include "surfel_fusion_common.hpp"

#include "camera.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

template <Device dev>
struct LiveMergeKernel {
  const IndexMapAccessor<dev> target_indexmap;
  const IndexMapAccessor<dev> live_indexmap;
  SurfelModelAccessor<dev> surfel_model;
  const RTCamera<dev, float> rt_cam;
  int scale, search_size;
  float max_normal_angle;

  typename Accessor<dev, int64_t, 2>::T new_surfel_map;

  LiveMergeKernel(const IndexMap &target_indexmap,
                  const IndexMap &live_indexmap,
                  const MappedSurfelModel &surfel_model,
                  RTCamera<dev, float> rt_cam, int search_size,
                  float max_normal_angle, torch::Tensor new_surfel_map)
      : target_indexmap(target_indexmap),
        live_indexmap(live_indexmap),
        surfel_model(surfel_model),
        rt_cam(rt_cam),
        search_size(search_size),
        max_normal_angle(max_normal_angle),
        new_surfel_map(Accessor<dev, int64_t, 2>::Get(new_surfel_map)) {
    scale = int(float(target_indexmap.get_height()) / live_indexmap.get_height());
    search_size = int(scale * search_size);
  }

  FTB_DEVICE_HOST void operator()(int row, int col) {
    new_surfel_map[row][col] = -1;
    if (live_indexmap.empty(row, col)) return;

    const Vector<float, 3> live_pos(live_indexmap.position(row, col));
    const float lambda =
        sqrt(live_pos[0] * live_pos[0] + live_pos[1] * live_pos[1] + 1);

    const Vector<float, 3> live_normal(live_indexmap.normal(row, col));

    const int xstart = max(col * scale - search_size, 0);
    const int xend =
        min(col * scale + search_size, int(target_indexmap.height()) - 1);

    const int ystart = max(row * scale - search_size, 0);
    const int yend =
        min(row * scale + search_size, int(target_indexmap.width()) - 1);

    float best_dist = NumericLimits<dev, float>::infinity();
    int best = -1;

    for (int krow = ystart; krow <= yend; krow++) {
      for (int kcol = xstart; kcol <= xend; kcol++) {
        if (target_indexmap.empty(krow, kcol)) continue;

        const int current = target_indexmap.index(krow, kcol);

        const Vector<float, 3> model_pos = target_indexmap.position(krow, kcol);
        if (abs((model_pos[2] * lambda) - (live_pos[2] * lambda)) >= 0.05)
          continue;

        const float dist =
            live_pos.cross(model_pos).squaredNorm() / live_pos.squaredNorm();

        const Vector<float, 3> normal = target_indexmap.normal(krow, kcol);
        if (dist < best_dist &&
            (abs(normal[2]) < 0.75f ||
             abs(GetVectorsAngle(normal, live_normal)) < max_normal_angle)) {
          best_dist = dist;
          best = current;
        }
      }
    }

    if (best >= 0) {
      const float live_conf = live_indexmap.confidence(row, col);
      const float model_conf = surfel_model.confidences[best];
      const float conf_total = live_conf + model_conf;

      const Vector<float, 3> live_world_pos = rt_cam.Transform(live_pos);
      surfel_model.set_position(best,
                                (surfel_model.position(best) * model_conf +
                                 live_world_pos * live_conf) /
                                    conf_total);
      surfel_model.set_normal(best, (surfel_model.normal(best) * model_conf +
                                     live_normal * live_conf) /
                                        conf_total);

      const Vector<float, 3> live_color(live_indexmap.color(row, col));
      surfel_model.set_color(best, (surfel_model.color(best) * model_conf +
                                    live_color * live_conf) /
                                       conf_total);
      surfel_model.confidences[best] = conf_total;
    } else {
      new_surfel_map[row][col] = live_indexmap.index(row, col);
    }
  }
};

}  // namespace

void SurfelFusionOp::MergeLive(const IndexMap &target_indexmap,
                               const IndexMap &live_indexmap,
                               const MappedSurfelModel &model,
                               const torch::Tensor &rt_cam, int search_size,
                               float max_normal_angle,
                               torch::Tensor new_surfels_map) {
  auto reference_dev = target_indexmap.get_device();

  live_indexmap.CheckDevice(reference_dev);
  target_indexmap.CheckDevice(reference_dev);
  model.CheckDevice(reference_dev);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);

  if (reference_dev.is_cuda()) {
    LiveMergeKernel<kCUDA> kernel(target_indexmap, live_indexmap, model,
                                  RTCamera<kCUDA, float>(rt_cam), search_size,
                                  max_normal_angle, new_surfels_map);
    Launch2DKernelCUDA(kernel, live_indexmap.get_width(), live_indexmap.get_height());
  } else {
    LiveMergeKernel<kCPU> kernel(target_indexmap, live_indexmap, model,
                                 RTCamera<kCPU, float>(rt_cam), search_size,
                                 max_normal_angle, new_surfels_map);
    Launch2DKernelCPU(kernel, live_indexmap.get_width(), live_indexmap.get_height());
  }
}
}  // namespace fiontb