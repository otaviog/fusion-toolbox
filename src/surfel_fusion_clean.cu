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
struct CleanKernel {
  SurfelModelAccessor<dev> model;
  const typename Accessor<dev, int64_t, 1>::T model_indices;

  const int time;
  const int max_time_thresh;
  const float stable_conf_thresh;

  typename Accessor<dev, bool, 1>::T remove_mask;

  CleanKernel(MappedSurfelModel model, torch::Tensor model_indices, int time,
              int max_time_thresh, float stable_conf_thresh,
              torch::Tensor remove_mask)
      : model(model),
        model_indices(Accessor<dev, int64_t, 1>::Get(model_indices)),
        time(time),
        max_time_thresh(max_time_thresh),
        stable_conf_thresh(stable_conf_thresh),
        remove_mask(Accessor<dev, bool, 1>::Get(remove_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    int64_t model_idx = model_indices[idx];

    if (model.confidences[model_idx] < stable_conf_thresh &&
        (time - model.times[model_idx]) > max_time_thresh) {
      remove_mask[idx] = true;
      return;
    }
  }
};

}  // namespace

void SurfelFusionOp::Clean(MappedSurfelModel model, torch::Tensor model_indices,
                           int time,  int stable_time_thresh,
						   float stable_conf_thresh,
						   torch::Tensor remove_mask) {
  const auto ref_device = model_indices.device();
  model.CheckDevice(ref_device);
  FTB_CHECK_DEVICE(ref_device, remove_mask);

  if (ref_device.is_cuda()) {
    CleanKernel<kCUDA> kernel(model, model_indices, time, stable_time_thresh,
                              stable_conf_thresh, remove_mask);
    Launch1DKernelCUDA(kernel, model_indices.size(0));
  } else {
    CleanKernel<kCUDA> kernel(model, model_indices, time, stable_time_thresh,
                              stable_conf_thresh, remove_mask);
    Launch1DKernelCPU(kernel, model_indices.size(0));
  }
}
}  // namespace slamtb
