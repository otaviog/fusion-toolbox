#include "matching.hpp"

#include "camera.hpp"
#include "error.hpp"
#include "feature_map.hpp"
#include "kernel.hpp"

#include "merge_map.hpp"

namespace slamtb {
namespace {

template <Device dev, typename scalar_t>
struct ForwardKernel {
  const typename Accessor<dev, scalar_t, 3>::T target_points;
  const typename Accessor<dev, scalar_t, 3>::T target_normals;

  const FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T source_points;
  const KCamera<dev, scalar_t> kcam;
  const MergeMapAccessor<dev> merge_map;

  typename Accessor<dev, scalar_t, 2>::T out_points;
  typename Accessor<dev, scalar_t, 2>::T out_normals;
  typename Accessor<dev, scalar_t, 2>::T out_features;
  typename Accessor<dev, bool, 1>::T match_mask;

  ForwardKernel(const torch::Tensor &target_points,
                const torch::Tensor &target_normals,
                const torch::Tensor &target_features,
                const torch::Tensor &source_points, const torch::Tensor &kcam,
                const torch::Tensor &merge_map, torch::Tensor out_points,
                torch::Tensor out_normals, torch::Tensor out_features,
                torch::Tensor match_mask)
      : target_points(Accessor<dev, scalar_t, 3>::Get(target_points)),
        target_normals(Accessor<dev, scalar_t, 3>::Get(target_normals)),
        tgt_featmap(target_features),
        source_points(Accessor<dev, scalar_t, 2>::Get(source_points)),
        kcam(kcam),
        merge_map(merge_map),
        out_points(Accessor<dev, scalar_t, 2>::Get(out_points)),
        out_normals(Accessor<dev, scalar_t, 2>::Get(out_normals)),
        out_features(Accessor<dev, scalar_t, 2>::Get(out_features)),
        match_mask(Accessor<dev, bool, 1>::Get(match_mask)) {}

  STB_DEVICE_HOST void operator()(int source_idx) {
    match_mask[source_idx] = false;

    const Vector<scalar_t, 3> src_point =
        to_vec3<scalar_t>(source_points[source_idx]);
    scalar_t u, v;
    kcam.Project(src_point, u, v);

    const int ui = int(round(u));
    const int vi = int(round(v));
    if (ui < 0 || ui >= tgt_featmap.width || vi < 0 || vi >= tgt_featmap.height)
      return;

    const int32_t target_index = merge_map(vi, ui);
    if (target_index != source_idx) {
      return;
    }

    match_mask[source_idx] = true;

    const Vector<scalar_t, 3> tgt_point =
        to_vec3<scalar_t>(target_points[vi][ui]);
    const Vector<scalar_t, 3> tgt_normal =
        to_vec3<scalar_t>(target_normals[vi][ui]);

    out_points[source_idx][0] = tgt_point[0];
    out_points[source_idx][1] = tgt_point[1];
    out_points[source_idx][2] = tgt_point[2];

    out_normals[source_idx][0] = tgt_normal[0];
    out_normals[source_idx][1] = tgt_normal[1];
    out_normals[source_idx][2] = tgt_normal[2];

    const BilinearInterp<dev, scalar_t> interp = tgt_featmap.GetBilinear(u, v);
    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      scalar_t val = interp.Get(channel);
      out_features[channel][source_idx] = val;
    }
  }
};

}  // namespace

void FPCLMatcherOp::Forward(
    const torch::Tensor &target_points, const torch::Tensor &target_normals,
    const torch::Tensor &target_features, const torch::Tensor &source_points,
    const torch::Tensor &kcam, const torch::Tensor &merge_map,
    torch::Tensor out_points, torch::Tensor out_normals,
    torch::Tensor out_features, torch::Tensor match_mask) {
  const auto reference_dev = target_points.device();
  STB_CHECK_DEVICE(reference_dev, target_points);
  STB_CHECK_DEVICE(reference_dev, target_normals);
  STB_CHECK_DEVICE(reference_dev, target_features);
  STB_CHECK_DEVICE(reference_dev, source_points);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, out_points);
  STB_CHECK_DEVICE(reference_dev, out_normals);
  STB_CHECK_DEVICE(reference_dev, out_features);
  STB_CHECK_DEVICE(reference_dev, match_mask);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        target_points.scalar_type(), "FPCLMatcherOp::Forward", ([&] {
          ForwardKernel<kCUDA, scalar_t> kernel(
              target_points, target_normals, target_features, source_points,
              kcam, merge_map, out_points, out_normals, out_features,
              match_mask);

          Launch1DKernelCUDA(kernel, source_points.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        target_points.scalar_type(), "FPCLMatcherOp::Forward", ([&] {
          ForwardKernel<kCPU, scalar_t> kernel(
              target_points, target_normals, target_features, source_points,
              kcam, merge_map, out_points, out_normals, out_features,
              match_mask);

          Launch1DKernelCPU(kernel, source_points.size(0));
        }));
  }
}

namespace {
template <Device dev, typename scalar_t>
struct BackwardKernel {
  const FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T source_points;
  const typename Accessor<dev, bool, 1>::T match_mask;

  const typename Accessor<dev, scalar_t, 2>::T dl_feature;

  const KCamera<dev, scalar_t> kcam;
  const scalar_t grad_precision;

  typename Accessor<dev, scalar_t, 2>::T dx_points;

  BackwardKernel(const torch::Tensor &tgt_featmap,
                 const torch::Tensor &source_points,
                 const torch::Tensor &match_mask,
                 const torch::Tensor &dl_feature, const torch::Tensor &kcam,
                 scalar_t grad_precision, torch::Tensor dx_points)
      : tgt_featmap(tgt_featmap),
        source_points(Accessor<dev, scalar_t, 2>::Get(source_points)),
        match_mask(Accessor<dev, bool, 1>::Get(match_mask)),
        dl_feature(Accessor<dev, scalar_t, 2>::Get(dl_feature)),
        kcam(kcam),
        grad_precision(grad_precision),
        dx_points(Accessor<dev, scalar_t, 2>::Get(dx_points)) {}

  STB_DEVICE_HOST void operator()(int idx) {
    if (!match_mask[idx]) {
      dx_points[idx][0] = 0;
      dx_points[idx][1] = 0;
      dx_points[idx][2] = 0;
      return;
    }
    const Vector<scalar_t, 3> src_point = to_vec3<scalar_t>(source_points[idx]);

    scalar_t u, v;
    kcam.Project(src_point, u, v);

    BilinearInterpGrad<dev, scalar_t> grad =
        tgt_featmap.GetBilinearGrad(u, v, grad_precision);

    scalar_t dl_ugrad = 0;
    scalar_t dl_vgrad = 0;
    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      scalar_t du, dv;

      grad.Get(channel, du, dv);

      const scalar_t channel_dl = dl_feature[channel][idx];

      dl_ugrad += du * channel_dl;
      dl_vgrad += dv * channel_dl;
    }

    scalar_t j00, j02, j11, j12;
    kcam.Dx_Project(src_point, j00, j02, j11, j12);

    dx_points[idx][0] = j00 * dl_ugrad;
    dx_points[idx][1] = j11 * dl_vgrad;
    dx_points[idx][2] = j02 * dl_ugrad + j12 * dl_vgrad;
  }
};

}  // namespace

void FPCLMatcherOp::Backward(const torch::Tensor &target_features,
                             const torch::Tensor &source_points,
                             const torch::Tensor &match_mask,
                             const torch::Tensor &dl_feature,
                             const torch::Tensor &kcam, double grad_precision,
                             torch::Tensor dx_points) {
  const auto reference_dev = target_features.device();
  STB_CHECK_DEVICE(reference_dev, target_features);
  STB_CHECK_DEVICE(reference_dev, source_points);
  STB_CHECK_DEVICE(reference_dev, match_mask);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, dx_points);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        target_features.scalar_type(), "FPCLMatcherOp::Backward", ([&] {
          BackwardKernel<kCUDA, scalar_t> kernel(target_features, source_points,
                                                 match_mask, dl_feature, kcam,
                                                 grad_precision, dx_points);

          Launch1DKernelCUDA(kernel, source_points.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        target_features.scalar_type(), "FPCLMatcherOp::Backward", ([&] {
          BackwardKernel<kCPU, scalar_t> kernel(target_features, source_points,
                                                match_mask, dl_feature, kcam,
                                                grad_precision, dx_points);

          Launch1DKernelCPU(kernel, source_points.size(0));
        }));
  }
}

}  // namespace slamtb
