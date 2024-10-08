#include "icp_jacobian.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "correspondence.hpp"
#include "error.hpp"
#include "feature_map.hpp"
#include "icp_jacobian_common.hpp"
#include "merge_map.hpp"
#include "kernel.hpp"

namespace slamtb {
namespace {

template <Device dev, typename scalar_t, typename JacobianType,
          typename Correspondence>
struct FeatureJacobianKernel {
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;
  const RigidTransform<scalar_t> rt_cam;
  const FeatureMap<dev, scalar_t> tgt_featmap;
  const KCamera<dev, scalar_t> kcam;
  const MergeMapAccessor<dev> merge_map;
  const float residual_thresh;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jtr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residuals;

  AtomicInt<dev> match_count;

  FeatureJacobianKernel(
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &rt_cam,
      const torch::Tensor &tgt_featmap, const torch::Tensor &kcam,
      const torch::Tensor &merge_map, float residual_thresh,
      torch::Tensor JtJ_partial, torch::Tensor Jtr_partial,
      torch::Tensor squared_residuals, AtomicInt<dev> match_count)
      : src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        rt_cam(rt_cam),
        tgt_featmap(tgt_featmap),
        kcam(kcam),
        merge_map(merge_map),
        residual_thresh(residual_thresh),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jtr_partial(Accessor<dev, scalar_t, 2>::Get(Jtr_partial)),
        squared_residuals(Accessor<dev, scalar_t, 1>::Get(squared_residuals)),
        match_count(match_count) {}

#pragma nv_exec_check_disable
  STB_DEVICE_HOST void operator()(int source_idx) {
    JacobianType jacobian(JtJ_partial[source_idx], Jtr_partial[source_idx]);

    squared_residuals[source_idx] = 0.0;

    if (!src_mask[source_idx]) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[source_idx]));

    scalar_t u, v;
    kcam.Project(Tsrc_point, u, v);

    int ui = int(round(u));
    int vi = int(round(v));
    if (ui < 0 || ui >= tgt_featmap.width || vi < 0 || vi >= tgt_featmap.height)
      return;

    int32_t target_index = merge_map(vi, ui);
    if (target_index != source_idx) {
      return;
    }

    scalar_t squared_residual = 0;
    const BilinearInterp<dev, scalar_t> interp(tgt_featmap.GetBilinear(u, v));
    const BilinearInterpGrad<dev, scalar_t> dx_interp(
        tgt_featmap.GetBilinearGrad(u, v, 0.05));

    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      const scalar_t feat_residual =
          (interp.Get(channel) - src_feats[channel][source_idx]);
      scalar_t du, dv;

      dx_interp.Get(channel, du, dv);

      scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
      kcam.Dx_Project(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);

      const Vector<scalar_t, 3> gradk(du * j00_proj, dv * j11_proj,
                                      du * j02_proj + dv * j12_proj);

      jacobian.Compute(Tsrc_point, gradk, feat_residual);
      squared_residual += feat_residual * feat_residual;
    }

    if (squared_residual > residual_thresh * residual_thresh) {
      jacobian.Zero();
      return;
    }
    squared_residuals[source_idx] = squared_residual;
    ++match_count;
  }
};
}  // namespace

int ICPJacobian::EstimateFeature(
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &rt_cam,
    const torch::Tensor &tgt_feats, const torch::Tensor &kcam,
    const torch::Tensor merge_map, float residual_thresh,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residuals) {
  const auto reference_dev = src_points.device();

  STB_CHECK_DEVICE(reference_dev, src_feats);
  STB_CHECK_DEVICE(reference_dev, src_mask);
  STB_CHECK_DEVICE(reference_dev, rt_cam);
  STB_CHECK_DEVICE(reference_dev, tgt_feats);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, merge_map);

  STB_CHECK_DEVICE(reference_dev, JtJ_partial);
  STB_CHECK_DEVICE(reference_dev, Jr_partial);
  STB_CHECK_DEVICE(reference_dev, squared_residuals);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", ([&] {
          typedef RobustCorrespondence<kCUDA, scalar_t> Correspondence;
          typedef SE3ICPJacobian<kCUDA, scalar_t> TGroup;

          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, TGroup, Correspondence> kernel(
              src_points, src_feats, src_mask, rt_cam, tgt_feats, kcam,
              merge_map, residual_thresh, JtJ_partial, Jr_partial,
              squared_residuals, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", [&] {
          typedef RobustCorrespondence<kCPU, scalar_t> Correspondence;
          typedef SE3ICPJacobian<kCPU, scalar_t> TGroup;

          ScopedAtomicInt<kCPU> match_count;

          FeatureJacobianKernel<kCPU, scalar_t, TGroup, Correspondence> kernel(
              src_points, src_feats, src_mask, rt_cam, tgt_feats, kcam,
              merge_map, residual_thresh, JtJ_partial, Jr_partial,
              squared_residuals, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

int ICPJacobian::EstimateFeatureSO3(
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &rt_cam,
    const torch::Tensor &tgt_feats, const torch::Tensor &kcam,
    const torch::Tensor merge_map, float residual_thresh,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residuals) {
  const auto reference_dev = src_points.device();

  STB_CHECK_DEVICE(reference_dev, src_feats);
  STB_CHECK_DEVICE(reference_dev, src_mask);
  STB_CHECK_DEVICE(reference_dev, rt_cam);
  STB_CHECK_DEVICE(reference_dev, tgt_feats);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, merge_map);

  STB_CHECK_DEVICE(reference_dev, JtJ_partial);
  STB_CHECK_DEVICE(reference_dev, Jr_partial);
  STB_CHECK_DEVICE(reference_dev, squared_residuals);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", ([&] {
          typedef RobustCorrespondence<kCUDA, scalar_t> Correspondence;
          typedef SO3ICPJacobian<kCUDA, scalar_t> TGroup;

          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, TGroup, Correspondence> kernel(
              src_points, src_feats, src_mask, rt_cam, tgt_feats, kcam,
              merge_map, residual_thresh, JtJ_partial, Jr_partial,
              squared_residuals, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", [&] {
          typedef RobustCorrespondence<kCPU, scalar_t> Correspondence;
          typedef SO3ICPJacobian<kCPU, scalar_t> TGroup;

          ScopedAtomicInt<kCPU> match_count;

          FeatureJacobianKernel<kCPU, scalar_t, TGroup, Correspondence> kernel(
              src_points, src_feats, src_mask, rt_cam, tgt_feats, kcam,
              merge_map, residual_thresh, JtJ_partial, Jr_partial,
              squared_residuals, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }

  return num_matches;
}

}  // namespace slamtb
