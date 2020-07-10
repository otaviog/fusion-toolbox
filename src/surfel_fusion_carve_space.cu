#include "elastic_fusion.hpp"

#include <torch/torch.h>

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "surfel_fusion_common.hpp"

namespace slamtb {
namespace {

template <Device dev>
struct CarveKernel {
  SurfelModelAccessor<dev> model;
  const typename Accessor<dev, int64_t, 1>::T model_indices;
  const IndexMapAccessor<dev> model_indexmap;

  const KCamera<dev, float> kcam;
  const RigidTransform<float> world_to_cam;

  const int time;
  const int neighbor_size;
  const float stable_conf_thresh;
  const int stable_time_thresh;
  const float min_z_diff;

  typename Accessor<dev, bool, 2>::T indexmap_remove_mask;

  CarveKernel(MappedSurfelModel model, torch::Tensor model_indices,
              const IndexMap &model_indexmap, const torch::Tensor &kcam,
              const torch::Tensor &world_to_cam, int time, int neighbor_size,
              float stable_conf_thresh, int stable_time_thresh,
              float min_z_diff, torch::Tensor indexmap_remove_mask)
      : model(model),
        model_indices(Accessor<dev, int64_t, 1>::Get(model_indices)),
        model_indexmap(model_indexmap),
        kcam(kcam),
        world_to_cam(world_to_cam),
        time(time),
        neighbor_size(neighbor_size),
        stable_conf_thresh(stable_conf_thresh),
        stable_time_thresh(stable_time_thresh),
        min_z_diff(min_z_diff),
        indexmap_remove_mask(
            Accessor<dev, bool, 2>::Get(indexmap_remove_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const int64_t current_idx = model_indices[idx];

    if (model.confidences[current_idx] < stable_conf_thresh ||
        model.times[current_idx] != time) {
      // Not stable or recent merged.
      return;
    }

    const Eigen::Vector3f model_cam_point =
        world_to_cam.Transform(model.point(current_idx));
    if (model_cam_point[2] <= 0) return;

    int u, v;
    kcam.Projecti(model_cam_point, u, v);

    const int width = model_indexmap.width();
    const int height = model_indexmap.height();
    if (u < 0 || u >= width || v < 0 || v >= height) return;

#if 0    
    const Eigen::Vector3f model_local_normal =
        world_to_cam.TransformNormal(model.normal(current_idx));

    if (abs(model_local_normal[2]) > 0.85) {
      return;
    }
#endif

    const int start_row = max(v - neighbor_size, 0);
    const int end_row = min(v + neighbor_size, height - 1);
    const int start_col = max(u - neighbor_size, 0);
    const int end_col = min(u + neighbor_size, width - 1);

    for (int krow = start_row; krow <= end_row; ++krow) {
      for (int kcol = start_col; kcol <= end_col; ++kcol) {
        if (model_indexmap.empty(krow, kcol) ||
            current_idx == model_indexmap.index(krow, kcol))
          continue;

        const Eigen::Vector3f neighbor_normal(
            model_indexmap.normal_radius[krow][kcol][0],
            model_indexmap.normal_radius[krow][kcol][1],
            model_indexmap.normal_radius[krow][kcol][2]);

        if (neighbor_normal[2] < 0.0) { // ignore backfacing surfels
          continue;
        }

        const float neighbor_z = model_indexmap.point_confidence[krow][kcol][2];
        if (model_cam_point[2] - neighbor_z > min_z_diff) {
          indexmap_remove_mask[krow][kcol] = true;
        }
      }
    }
  }
};

}  // namespace

void SurfelFusionOp::CarveSpace(MappedSurfelModel model,
                                torch::Tensor model_indices,
                                const IndexMap &model_indexmap,
                                const torch::Tensor kcam,
                                const torch::Tensor &world_to_cam, int time,
                                int neighbor_size, float stable_conf_thresh,
                                int stable_time_thresh, float min_z_diff,
                                torch::Tensor indexmap_remove_mask) {
  const auto ref_device = model_indexmap.get_device();
  model.CheckDevice(ref_device);
  model_indexmap.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, kcam);
  FTB_CHECK_DEVICE(ref_device, indexmap_remove_mask);

  if (ref_device.is_cuda()) {
    CarveKernel<kCUDA> kernel(model, model_indices, model_indexmap, kcam,
                              world_to_cam, time, neighbor_size,
                              stable_conf_thresh, stable_time_thresh,
                              min_z_diff, indexmap_remove_mask);
    Launch1DKernelCUDA(kernel, model_indices.size(0));
  } else {
    CarveKernel<kCPU> kernel(model, model_indices, model_indexmap, kcam,
                             world_to_cam, time, neighbor_size,
                             stable_conf_thresh, stable_conf_thresh, min_z_diff,
                             indexmap_remove_mask);
    Launch1DKernelCPU(kernel, model_indices.size(0));
  }
}

}  // namespace slamtb
