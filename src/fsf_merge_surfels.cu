#include "fsf.hpp"

#include "surfel_fusion_common.hpp"

#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {
template <Device dev>
struct MergeKernel {
  const typename Accessor<dev, int64_t, 2>::T knn_index;
  const SurfelModelAccessor<dev> local_model;

  const typename Accessor<dev, int64_t, 1>::T global_map;
  SurfelModelAccessor<dev> global_model;

  const int search_size;

  MergeKernel(const torch::Tensor &knn_index, const SurfelCloud &local_model,
              const torch::Tensor &global_map,
              const MappedSurfelModel &global_model)
      : knn_index(Accessor<dev, int64_t, 2>::Get(knn_index)),
        local_model(local_model),
        global_map(Accessor<dev, int64_t, 1>::Get(global_map)),
        global_model(global_model),
        search_size(global_map.size(0)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const int64_t local_target_idx = knn_index[idx][0];
    if (local_target_idx >= search_size) return;

    const int64_t target_idx = global_map[local_target_idx];

    global_model.set_position(
        target_idx,
        (global_model.position(target_idx) + local_model.position(idx)) * .5);
  }
};

}  // namespace

void FSFOp::Merge(const torch::Tensor &knn_index,
                  const SurfelCloud &local_model,
                  const torch::Tensor &global_map,
                  MappedSurfelModel global_model) {
  const auto ref_device = knn_index.device();
  local_model.CheckDevice(ref_device);
  FTB_CHECK_DEVICE(ref_device, global_map);
  global_model.CheckDevice(ref_device);
  if (ref_device.is_cuda()) {
    MergeKernel<kCUDA> kernel(knn_index, local_model, global_map, global_model);
    Launch1DKernelCUDA(kernel, knn_index.size(0));
  } else {
    MergeKernel<kCPU> kernel(knn_index, local_model, global_map, global_model);
    Launch1DKernelCPU(kernel, knn_index.size(0));
  }
}

};  // namespace fiontb
