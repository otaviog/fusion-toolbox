#include "surfel_fusion_common.hpp"

#include <torch/torch.h>

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {

const int MAX_MERGE_VIOLATIONS = 1;

template <Device dev>
struct CleanKernel {
  SurfelModelAccessor<dev> model;
  const typename Accessor<dev, int64_t, 1>::T model_indices;
  const IndexMapAccessor<dev> model_indexmap;

  const KCamera<dev, float> kcam;
  const RTCamera<dev, float> inverse_rt;
  const Eigen::Matrix3f inverse_rt_normal;

  const int time;
  const int max_time_thresh;
  const int neighbor_size;
  const float stable_conf_thresh;

  typename Accessor<dev, bool, 1>::T remove_mask;

  CleanKernel(MappedSurfelModel model, torch::Tensor model_indices,
              const IndexMap &model_indexmap, const torch::Tensor &kcam,
              const torch::Tensor &inverse_rt,
              const Eigen::Matrix3f &inverse_rt_normal, int time,
              int max_time_thresh, int neighbor_size, float stable_conf_thresh,
              torch::Tensor remove_mask)
      : model(model),
        model_indices(Accessor<dev, int64_t, 1>::Get(model_indices)),
        model_indexmap(model_indexmap),
        kcam(kcam),
        inverse_rt(inverse_rt),
        inverse_rt_normal(inverse_rt_normal),
        time(time),
        max_time_thresh(max_time_thresh),
        neighbor_size(neighbor_size),
        stable_conf_thresh(stable_conf_thresh),
        remove_mask(Accessor<dev, bool, 1>::Get(remove_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    int64_t model_idx = model_indices[idx];

    if (model.confidences[model_idx] < stable_conf_thresh &&
        (time - model.times[model_idx]) > max_time_thresh) {
      remove_mask[idx] = true;
      return;
    }

    const Eigen::Vector3f model_xyz = model.position(model_idx);
    const Eigen::Vector3f model_normal = model.normal(model_idx);

    const Eigen::Vector3f model_local_xyz = inverse_rt.Transform(model_xyz);
    const Eigen::Vector3f model_local_normal = inverse_rt_normal * model_normal;

    const float model_radius = model.radii[model_idx];
    const int model_time = model.times[model_idx];

    int u, v;
    kcam.Projecti(model_local_normal, u, v);

    const int width = model_indexmap.width();
    const int height = model_indexmap.height();
    if (u < 0 || u >= width || v < 0 || v >= height) return;

    if (model_indexmap.empty(v, u)) return;
    if (model_indexmap.confidence(v, u) < stable_conf_thresh) return;

    const int start_row = max(v - neighbor_size, 0);
    const int end_row = min(v + neighbor_size, height - 1);

    const int start_col = max(u - neighbor_size, 0);
    const int end_col = min(u + neighbor_size, width - 1);

    int merge_count = 0;
    int carve_count = 0;

    for (int krow = start_row; krow <= end_row; ++krow) {
      for (int kcol = start_col; kcol <= end_col; ++kcol) {
        if (model_indexmap.empty(krow, kcol)) continue;
        if (model_indexmap.confidence(krow, kcol) < stable_conf_thresh)
          continue;

        const Eigen::Vector3f im_xyz = model_indexmap.position(krow, kcol);
        const Eigen::Vector3f im_normal = model_indexmap.normal(krow, kcol);
        const float im_conf = model_indexmap.confidence(krow, kcol);

        if (im_conf > stable_conf_thresh &&
            im_xyz[2] > model_local_xyz[2] &
                im_xyz[2] - model_local_xyz[2] < 0.01 &&
            Eigen::Vector2f(im_xyz[0] - model_local_xyz[0],
                            im_xyz[1] - model_local_xyz[0])
                    .norm() < model_radius * 1.4) {
          ++merge_count;
        }

        if (model_time == time && im_conf > stable_conf_thresh &&
            im_xyz[2] > model_local_xyz[2] &&
            im_xyz[2] - model_local_xyz[2] > 0.01 &&
            abs(model_local_xyz[2]) > 0.85) {
          ++carve_count;
        }
      }
    }

    if (merge_count > 8 || carve_count > 4) {
      // Remove
      remove_mask[idx] = true;
    }
  }
};

}  // namespace

void SurfelFusionOp::Clean(MappedSurfelModel model, torch::Tensor model_indices,
                           const IndexMap &model_indexmap,
                           const torch::Tensor &kcam,
                           const torch::Tensor &world_to_cam, int time,
                           int max_time_thresh, int neighbor_size,
                           float stable_conf_thresh,
                           torch::Tensor remove_mask) {
  const auto ref_device = model_indexmap.get_device();
  model.CheckDevice(ref_device);
  model_indexmap.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, kcam);
  FTB_CHECK_DEVICE(ref_device, remove_mask);

  const auto world_to_cam_cpu = world_to_cam.cpu();
  Eigen::Matrix4f inverse_rt_cam(
      to_matrix<float, 4, 4>(world_to_cam_cpu.accessor<float, 2>()));
  Eigen::Matrix3f inverse_rt_normal(inverse_rt_cam.topLeftCorner(3, 3));
  inverse_rt_normal = inverse_rt_normal.inverse().transpose();
  if (ref_device.is_cuda()) {
    CleanKernel<kCUDA> kernel(model, model_indices, model_indexmap, kcam,
                              world_to_cam, inverse_rt_normal, time,
                              max_time_thresh, neighbor_size,
                              stable_conf_thresh, remove_mask);
    Launch1DKernelCUDA(kernel, model_indices.size(0));
  } else {
    CleanKernel<kCUDA> kernel(model, model_indices, model_indexmap, kcam,
                              world_to_cam, inverse_rt_normal, time,
                              max_time_thresh, neighbor_size,
                              stable_conf_thresh, remove_mask);
    Launch1DKernelCPU(kernel, model_indices.size(0));
  }
}
}  // namespace fiontb
