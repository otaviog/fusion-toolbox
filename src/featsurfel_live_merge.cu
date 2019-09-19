#include "featsurfel.hpp"

#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

template <Device dev>
struct LiveMergeKernel {
  const IndexMap<dev> target_indexmap;
  const IndexMap<dev> live_indexmap;
  SurfelModel<dev> surfel_model;

  int scale, search_size;
  float max_normal_angle;

  LiveMergeKernel(IndexMap<dev> target_indexmap, IndexMap<dev> live_indexmap,
                  SurfelModel<dev> surfel_model, int search_size,
                  float max_normal_angle)
      : target_indexmap(target_indexmap),
        live_indexmap(live_indexmap),
        surfel_model(surfel_model),
        max_normal_angle(max_normal_angle) {
    scale = int(float(target_indexmap.height()) / live_indexmap.height());
    search_size = int(scale * search_size);
  }

  FTB_DEVICE_HOST void operator()(int row, int col) {
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
  }
};

}  // namespace

void FeatSurfel::MergeLive(const IndexMapParams &target_indexmap_params,
                           const IndexMapParams &live_indexmap_params,
                           const SurfelModelParams &model_params,
                           int search_size, float max_normal_angle) {
  if (live_indexmap_params.IsCuda()) {
    live_indexmap_params.CheckCuda();
    target_indexmap_params.CheckCuda();
    model_params.CheckCuda();

    IndexMap<kCUDA> live_indexmap(live_indexmap_params);
    IndexMap<kCUDA> target_indexmap(target_indexmap_params);
    SurfelModel<kCUDA> model(model_params);

    LiveMergeKernel<kCUDA> kernel(target_indexmap, live_indexmap, model,
                                  search_size, max_normal_angle);
    Launch2DKernelCUDA(kernel, live_indexmap.width(), live_indexmap.height());
  } else {
    live_indexmap_params.CheckCpu();
    target_indexmap_params.CheckCpu();
    model_params.CheckCpu();

    IndexMap<kCPU> live_indexmap(live_indexmap_params);
    IndexMap<kCPU> target_indexmap(target_indexmap_params);
    SurfelModel<kCPU> model(model_params);

    LiveMergeKernel<kCPU> kernel(target_indexmap, live_indexmap, model,
                                 search_size, max_normal_angle);
    Launch2DKernelCPU(kernel, live_indexmap.width(), live_indexmap.height());
  }
}
}  // namespace fiontb