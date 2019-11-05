#include "elastic_fusion.hpp"

#include <torch/torch.h>

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "surfel_fusion_common.hpp"

namespace fiontb {
namespace {

template <Device dev>
struct CarveKernel {
  SurfelModelAccessor<dev> model;
  const typename Accessor<dev, int64_t, 1>::T model_indices;
  const IndexMapAccessor<dev> model_indexmap;

  const KCamera<dev, float> kcam;
  const RTCamera<float> inverse_rt;

  const int time;
  const int neighbor_size;
  const float stable_conf_thresh;
  const int stable_time_thresh;
  const float min_z_diff;

  typename Accessor<dev, bool, 1>::T remove_mask;

  CarveKernel(MappedSurfelModel model, torch::Tensor model_indices,
              const IndexMap &model_indexmap, const torch::Tensor &kcam,
              const torch::Tensor &inverse_rt, int time, int neighbor_size,
              float stable_conf_thresh, int stable_time_thresh,
              torch::Tensor remove_mask)
      : model(model),
        model_indices(Accessor<dev, int64_t, 1>::Get(model_indices)),
        model_indexmap(model_indexmap),
        kcam(kcam),
        inverse_rt(inverse_rt),
        time(time),
        neighbor_size(neighbor_size),
        stable_conf_thresh(stable_conf_thresh),
        stable_time_thresh(stable_time_thresh),
        min_z_diff(0.01),
        remove_mask(Accessor<dev, bool, 1>::Get(remove_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const int64_t current_idx = model_indices[idx];

    if (time - model.times[current_idx] >= stable_time_thresh) return;

    const Eigen::Vector3f curr_xyz =
        inverse_rt.Transform(model.position(current_idx));
    const float curr_z = curr_xyz[2];

    if (curr_z <= 0) return;

    int u, v;
    kcam.Projecti(curr_xyz, u, v);

    const int width = model_indexmap.width();
    const int height = model_indexmap.height();
    if (u < 0 || u >= width || v < 0 || v >= height) return;

    const Eigen::Vector3f model_local_normal =
        inverse_rt.TransformNormal(model.normal(current_idx));
    if (abs(model_local_normal[2]) > 0.85) {
      return;
    }

    const int start_row = max(v - neighbor_size, 0);
    const int end_row = min(v + neighbor_size, height - 1);
    const int start_col = max(u - neighbor_size, 0);
    const int end_col = min(u + neighbor_size, width - 1);

    int violation_count = 0;

    for (int krow = start_row; krow <= end_row; ++krow) {
      for (int kcol = start_col; kcol <= end_col; ++kcol) {
        const float neighbor_conf = model_indexmap.confidence(krow, kcol);
        if (model_indexmap.empty(krow, kcol) ||
            current_idx == model_indexmap.index(krow, kcol) ||
            neighbor_conf < stable_conf_thresh)
          continue;

        const float neighbor_z =
            model_indexmap.position_confidence[krow][kcol][2];
        const int neighbor_time = model_indexmap.time(krow, kcol);

        if (neighbor_time == time && neighbor_conf > stable_conf_thresh &&
            neighbor_z > curr_z && neighbor_z - curr_z > min_z_diff) {
          ++violation_count;
        }
      }
    }

    if (violation_count > 4) {
      remove_mask[idx] = true;
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
                                int stable_time_thresh,
                                torch::Tensor remove_mask) {
  const auto ref_device = model_indexmap.get_device();
  model.CheckDevice(ref_device);
  model_indexmap.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, kcam);
  FTB_CHECK_DEVICE(ref_device, remove_mask);

  if (ref_device.is_cuda()) {
    CarveKernel<kCUDA> kernel(
        model, model_indices, model_indexmap, kcam, world_to_cam, time,
        neighbor_size, stable_conf_thresh, stable_time_thresh, remove_mask);
    Launch1DKernelCUDA(kernel, model_indices.size(0));
  } else {
    CarveKernel<kCPU> kernel(
        model, model_indices, model_indexmap, kcam, world_to_cam, time,
        neighbor_size, stable_conf_thresh, stable_conf_thresh, remove_mask);
    Launch1DKernelCPU(kernel, model_indices.size(0));
  }
}

}  // namespace fiontb
