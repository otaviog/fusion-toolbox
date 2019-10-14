#include "surfel_fusion_common.hpp"

#include <torch/torch.h>

#include "accessor.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {
template <Device dev>
struct CopyFeaturesKernel {
  const typename Accessor<dev, int32_t, 3>::T indexmap;
  const typename Accessor<dev, float, 2>::T model_features;
  typename Accessor<dev, float, 3>::T out_features;
  const bool flip;

  CopyFeaturesKernel(const torch::Tensor &indexmap,
                     const torch::Tensor &model_features,
                     torch::Tensor out_features, bool flip)
      : indexmap(Accessor<dev, int32_t, 3>::Get(indexmap)),
        model_features(Accessor<dev, float, 2>::Get(model_features)),
        out_features(Accessor<dev, float, 3>::Get(out_features)),
        flip(flip) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    if (!indexmap[row][col][1]) return;
    const int32_t surfel_index = indexmap[row][col][0];

    for (int64_t i = 0; i < out_features.size(0); ++i) {
      const float value = model_features[i][surfel_index];
	  
      if (flip)
        out_features[i][out_features.size(1) - 1 - row][col] = value;
      else
        out_features[i][row][col] = value;
    }
  }
};
}  // namespace

void SurfelFusionOp::CopyFeatures(const torch::Tensor &indexmap,
                                  const torch::Tensor &model_features,
                                  torch::Tensor out_features, bool flip) {
  const auto ref_device = indexmap.device();
  FTB_CHECK_DEVICE(ref_device, model_features);
  FTB_CHECK_DEVICE(ref_device, out_features);

  if (ref_device.is_cuda()) {
    CopyFeaturesKernel<kCUDA> kernel(indexmap, model_features, out_features,
                                     flip);
    Launch2DKernelCUDA(kernel, out_features.size(2), out_features.size(1));
  } else {
    CopyFeaturesKernel<kCPU> kernel(indexmap, model_features, out_features,
                                    flip);
    Launch2DKernelCPU(kernel, out_features.size(2), out_features.size(1));
  }
}
}  // namespace fiontb
