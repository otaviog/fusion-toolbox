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
#include "kernel.hpp"

namespace slamtb {
namespace {

template <Device dev, typename scalar_t, typename JacobianType,
          typename Correspondence>
struct FeatureJacobianKernel {
  const Correspondence corresp;
  const FeatureMap<dev, scalar_t> tgt_featmap;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const RigidTransform<scalar_t> rt_cam;

  const float residual_thresh;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jtr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residuals;

  AtomicInt<dev> match_count;

  FeatureJacobianKernel(const Correspondence &corresp,
                        const torch::Tensor &tgt_featmap,
                        const torch::Tensor &src_points,
                        const torch::Tensor &src_feats,
                        const torch::Tensor &src_mask,
                        const torch::Tensor &rt_cam, float residual_thresh,
                        torch::Tensor JtJ_partial, torch::Tensor Jtr_partial,
                        torch::Tensor squared_residuals,
                        AtomicInt<dev> match_count)
      : corresp(corresp),
        tgt_featmap(tgt_featmap),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        rt_cam(rt_cam),
        residual_thresh(residual_thresh),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jtr_partial(Accessor<dev, scalar_t, 2>::Get(Jtr_partial)),
        squared_residuals(Accessor<dev, scalar_t, 1>::Get(squared_residuals)),
        match_count(match_count) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int source_idx) {
    JacobianType jacobian(JtJ_partial[source_idx], Jtr_partial[source_idx]);

    squared_residuals[source_idx] = 0.0;

    if (src_mask[source_idx] == 0) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[source_idx]));
    
    scalar_t u, v;
    if (!corresp.Match(Tsrc_point, u, v)) {
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
      corresp.kcam.Dx_Project(Tsrc_point, j00_proj, j02_proj, j11_proj,
                              j12_proj);

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
} // namespace

int ICPJacobian::EstimateFeature(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_feats, const torch::Tensor &tgt_mask,
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &kcam,
    const torch::Tensor &rt_cam, float distance_thresh,
    float normals_angle_thresh, float residual_thresh,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residuals) {
  const auto reference_dev = src_points.device();
  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_feats);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_feats);
  FTB_CHECK_DEVICE(reference_dev, src_mask);

  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, JtJ_partial);
  FTB_CHECK_DEVICE(reference_dev, Jr_partial);
  FTB_CHECK_DEVICE(reference_dev, squared_residuals);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", ([&] {
          typedef RobustCorrespondence<kCUDA, scalar_t> Correspondence;
          typedef SE3ICPJacobian<kCUDA, scalar_t> TGroup;

          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, TGroup, Correspondence> kernel(
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam,
                             distance_thresh, normals_angle_thresh),
              tgt_feats, src_points, src_feats, src_mask, rt_cam,
              residual_thresh, JtJ_partial, Jr_partial, squared_residuals,
              match_count.get());

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
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam,
                             distance_thresh, normals_angle_thresh),
              tgt_feats, src_points, src_feats, src_mask, rt_cam,
              residual_thresh, JtJ_partial, Jr_partial, squared_residuals,
              match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

int ICPJacobian::EstimateFeatureSO3(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_feats, const torch::Tensor &tgt_mask,
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &kcam,
    const torch::Tensor &rt_cam, float distance_thresh,
    float normals_angle_thresh, float residual_thresh,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residuals) {
  const auto reference_dev = src_points.device();
  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_feats);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_feats);
  FTB_CHECK_DEVICE(reference_dev, src_mask);

  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, JtJ_partial);
  FTB_CHECK_DEVICE(reference_dev, Jr_partial);
  FTB_CHECK_DEVICE(reference_dev, squared_residuals);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", ([&] {
          typedef RobustCorrespondence<kCUDA, scalar_t> Correspondence;
          typedef SO3ICPJacobian<kCUDA, scalar_t> TGroup;

          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, TGroup, Correspondence> kernel(
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam,
                             distance_thresh, normals_angle_thresh),
              tgt_feats, src_points, src_feats, src_mask, rt_cam,
              residual_thresh, JtJ_partial, Jr_partial, squared_residuals,
              match_count.get());

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
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam,
                             distance_thresh, normals_angle_thresh),
              tgt_feats, src_points, src_feats, src_mask, rt_cam,
              residual_thresh, JtJ_partial, Jr_partial, squared_residuals,
              match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }

  return num_matches;
}

}  // namespace slamtb
