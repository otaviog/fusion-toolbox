#include "feature_map.hpp"
#include "filtering.hpp"

#include "camera.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
struct ForwardKernel {
  const FeatureMap<dev, scalar_t> feature_map;
  const typename Accessor<dev, scalar_t, 2>::T uv;
  typename Accessor<dev, scalar_t, 2>::T out_features;
  typename Accessor<dev, bool, 1>::T out_bound_mask;

  ForwardKernel(const torch::Tensor feature_map, const torch::Tensor uv,
                torch::Tensor out_features, torch::Tensor out_bound_mask)
      : feature_map(feature_map),
        uv(Accessor<dev, scalar_t, 2>::Get(uv)),
        out_features(Accessor<dev, scalar_t, 2>::Get(out_features)),
        out_bound_mask(Accessor<dev, bool, 1>::Get(out_bound_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    out_bound_mask[idx] = false;

    scalar_t u = uv[idx][0];
    scalar_t v = uv[idx][1];

    if (u < 0 || u >= feature_map.width || v < 0 || v >= feature_map.height)
      return;

    out_bound_mask[idx] = true;

    const BilinearInterp<dev, scalar_t> interp = feature_map.GetBilinear(u, v);
    for (int channel = 0; channel < feature_map.channel_size; ++channel) {
      scalar_t val = interp.Get(channel);
      out_features[channel][idx] = val;
    }
  }
};

}  // namespace

void FeatureMapOp::Forward(const torch::Tensor feature_map,
                           const torch::Tensor uv, torch::Tensor out_features,
                           torch::Tensor out_bound_mask) {
  FTB_CHECK_DEVICE(feature_map.device(), uv);
  FTB_CHECK_DEVICE(feature_map.device(), out_features);
  FTB_CHECK_DEVICE(feature_map.device(), out_bound_mask);

  if (feature_map.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        feature_map.scalar_type(), "MatchDensePoints", ([&] {
          ForwardKernel<kCUDA, scalar_t> kernel(feature_map, uv, out_features,
                                                out_bound_mask);
          Launch1DKernelCUDA(kernel, uv.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        feature_map.scalar_type(), "MatchDensePoints", ([&] {
          ForwardKernel<kCPU, scalar_t> kernel(feature_map, uv, out_features,
                                               out_bound_mask);
          Launch1DKernelCPU(kernel, uv.size(0));
        }));
  }
}

namespace {
template <Device dev, typename scalar_t>
struct BackwardKernel {
  const FeatureMap<dev, scalar_t> feature_map;
  const typename Accessor<dev, scalar_t, 2>::T uv;
  const typename Accessor<dev, scalar_t, 2>::T dl_value;
  typename Accessor<dev, scalar_t, 2>::T dl_uv;

  BackwardKernel(const torch::Tensor feature_map, const torch::Tensor uv,
                 const torch::Tensor dl_value, torch::Tensor dl_uv)
      : feature_map(feature_map),
        uv(Accessor<dev, scalar_t, 2>::Get(uv)),
        dl_value(Accessor<dev, scalar_t, 2>::Get(dl_value)),
        dl_uv(Accessor<dev, scalar_t, 2>::Get(dl_uv)) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int idx) {
    const scalar_t u = uv[idx][0];
    const scalar_t v = uv[idx][1];

    dl_uv[idx][0] = 0;
    dl_uv[idx][1] = 0;

    if (u < 0 || u >= feature_map.width || v < 0 || v >= feature_map.height)
      return;

    const scalar_t h = 0.05;

    scalar_t dl_ugrad = 0;
    scalar_t dl_vgrad = 0;

    BilinearInterpGrad<dev, scalar_t> grad =
        feature_map.GetBilinearGrad(u, v, h);

    for (int channel = 0; channel < feature_map.channel_size; ++channel) {
      scalar_t du, dv;
      grad.Get(channel, du, dv);

      const scalar_t channel_dl = dl_value[channel][idx];

      dl_ugrad += du * channel_dl;
      dl_vgrad += dv * channel_dl;
    }

    dl_uv[idx][0] = dl_ugrad;
    dl_uv[idx][1] = dl_vgrad;
  }
};
}  // namespace

void FeatureMapOp::Backward(const torch::Tensor feature_map,
                            const torch::Tensor uv,
                            const torch::Tensor dl_value, torch::Tensor dl_uv) {
  FTB_CHECK_DEVICE(feature_map.device(), uv);
  FTB_CHECK_DEVICE(feature_map.device(), dl_value);
  FTB_CHECK_DEVICE(feature_map.device(), dl_uv);

  if (feature_map.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(feature_map.scalar_type(), "MatchDensePoints",
                               ([&] {
                                 BackwardKernel<kCUDA, scalar_t> kernel(
                                     feature_map, uv, dl_value, dl_uv);
                                 Launch1DKernelCUDA(kernel, uv.size(0));
                               }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(feature_map.scalar_type(), "MatchDensePoints",
                               ([&] {
                                 BackwardKernel<kCPU, scalar_t> kernel(
                                     feature_map, uv, dl_value, dl_uv);
                                 Launch1DKernelCPU(kernel, uv.size(0));
                               }));
  }
}
}  // namespace fiontb