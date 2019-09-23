#include "featsurfel.hpp"

#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

template <Device dev>
struct LiveMergeKernel {
  const IndexMapAccessor<dev> target_indexmap;
  const IndexMapAccessor<dev> live_indexmap;
  SurfelModelAccessor<dev> surfel_model;

  int scale, search_size;
  float max_normal_angle;

  typename Accessor<dev, int64_t, 2>::T new_surfel_map;

  LiveMergeKernel(IndexMapAccessor<dev> target_indexmap,
                  IndexMapAccessor<dev> live_indexmap,
                  SurfelModelAccessor<dev> surfel_model, int search_size,
                  float max_normal_angle, torch::Tensor new_surfel_map)
      : target_indexmap(target_indexmap),
        live_indexmap(live_indexmap),
        surfel_model(surfel_model),
        max_normal_angle(max_normal_angle),
        new_surfel_map(Accessor<dev, int64_t, 2>::Get(new_surfel_map)) {
    scale = int(float(target_indexmap.height()) / live_indexmap.height());
    search_size = int(scale * search_size);
  }

  FTB_DEVICE_HOST void operator()(int row, int col) {
    new_surfel_map[row][col] = -1;
    if (live_indexmap.empty(row, col)) return;

    const Vector<float, 3> ray(live_indexmap.position(row, col));
    const float lambda = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1);

    const Vector<float, 3> view_normal(live_indexmap.normal(row, col));

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

        const Vector<float, 3> vert = target_indexmap.position(krow, kcol);
        if (abs((vert[2] * lambda) - (ray[2] * lambda)) >= 0.05) continue;

        const float dist = ray.cross(vert).norm() / ray.norm();
        const Vector<float, 3> normal = target_indexmap.normal(krow, kcol);

        if (dist < best_dist &&
            (abs(normal[2]) < 0.75f ||
             abs(GetVectorsAngle(normal, view_normal)) < max_normal_angle)) {
          best_dist = dist;
          best = current;
        }
      }
    }

    if (best >= 0) {
      const float live_conf = live_indexmap.confidence(row, col);
      const float model_conf = surfel_model.confidences[best];
      const float conf_total = live_conf + model_conf;

      surfel_model.set_position(
          best, (surfel_model.position(best) * model_conf + ray * conf_total) /
                    conf_total);
      surfel_model.set_normal(best, (surfel_model.normal(best) * model_conf +
                                     view_normal * live_conf) /
                                        conf_total);
      surfel_model.set_color(best, (surfel_model.color(best) * model_conf +
                                    view_normal * live_conf) /
                                       conf_total);
      surfel_model.confidences[best] = conf_total;
    } else {
      new_surfel_map[row][col] = live_indexmap.index(row, col);
    }
  }
};

}  // namespace

void FeatSurfel::MergeLive(const IndexMap &target_indexmap_params,
                           const IndexMap &live_indexmap_params,
                           const MappedSurfelModel &model_params,
                           int search_size, float max_normal_angle,
                           torch::Tensor new_surfels_map) {
  auto reference_dev = target_indexmap_params.get_device();
  live_indexmap_params.CheckDevice(reference_dev);
  target_indexmap_params.CheckDevice(reference_dev);
  model_params.CheckDevice(reference_dev);

  if (reference_dev.is_cuda()) {
    IndexMapAccessor<kCUDA> live_indexmap(live_indexmap_params);
    IndexMapAccessor<kCUDA> target_indexmap(target_indexmap_params);
    SurfelModelAccessor<kCUDA> model(model_params);

    LiveMergeKernel<kCUDA> kernel(target_indexmap, live_indexmap, model,
                                  search_size, max_normal_angle,
                                  new_surfels_map);
    Launch2DKernelCUDA(kernel, live_indexmap.width(), live_indexmap.height());
  } else {
    IndexMapAccessor<kCPU> live_indexmap(live_indexmap_params);
    IndexMapAccessor<kCPU> target_indexmap(target_indexmap_params);
    SurfelModelAccessor<kCPU> model(model_params);

    LiveMergeKernel<kCPU> kernel(target_indexmap, live_indexmap, model,
                                 search_size, max_normal_angle,
                                 new_surfels_map);
    Launch2DKernelCPU(kernel, live_indexmap.width(), live_indexmap.height());
  }
}
}  // namespace fiontb