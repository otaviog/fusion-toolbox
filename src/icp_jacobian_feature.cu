#include "icpodometry.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "error.hpp"
#include "icp_jacobian_common.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
struct SE3Jacobian {
  typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial;
  typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial;

  FTB_DEVICE_HOST SE3Jacobian(
      typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial,
      typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial)
      : JtJ_partial(JtJ_partial), Jtr_partial(Jtr_partial) {
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      Jtr_partial[k] = scalar_t(0);
    }

#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[krow][kcol] = scalar_t(0);
      }
    }
  }

  FTB_DEVICE_HOST inline void Compute(const Vector<scalar_t, 3> &Tsrc_point,
                                      const Vector<scalar_t, 3> &gradk,
                                      scalar_t residual) {
    scalar_t jacobian[6];
    jacobian[0] = gradk[0];
    jacobian[1] = gradk[1];
    jacobian[2] = gradk[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(gradk);
    jacobian[3] = rot_twist[0];
    jacobian[4] = rot_twist[1];
    jacobian[5] = rot_twist[2];

    for (int k = 0; k < 6; ++k) {
      Jtr_partial[k] = jacobian[k] * residual;
    }

#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[krow][kcol] = jacobian[kcol] * jacobian[krow];
      }
    }
  }
};

template <Device dev, typename scalar_t>
struct SO3Jacobian {
  typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial;
  typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial;

  FTB_DEVICE_HOST SO3Jacobian(
      typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial,
      typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial)
      : JtJ_partial(JtJ_partial), Jtr_partial(Jtr_partial) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      Jtr_partial[k] = scalar_t(0);
    }

#pragma unroll
    for (int krow = 0; krow < 3; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 3; ++kcol) {
        JtJ_partial[krow][kcol] = scalar_t(0);
      }
    }
  }

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST inline void Compute(const Vector<scalar_t, 3> &Tsrc_point,
                                      const Vector<scalar_t, 3> &gradk,
                                      scalar_t residual) {
    scalar_t jacobian[3];
    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(gradk);
    jacobian[0] = rot_twist[0];
    jacobian[1] = rot_twist[1];
    jacobian[2] = rot_twist[2];

    for (int k = 0; k < 3; ++k) {
      Jtr_partial[k] = jacobian[k] * residual;
    }

#pragma unroll
    for (int krow = 0; krow < 3; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 3; ++kcol) {
        JtJ_partial[krow][kcol] = jacobian[kcol] * jacobian[krow];
      }
    }
  }
};

template <Device dev, typename scalar_t, typename JacobianType>
struct FeatureJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const KCamera<dev, scalar_t> kcam;
  const RigidTransform<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jtr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  FeatureJacobianKernel(const PointGrid<dev, scalar_t> tgt,
                        FeatureMap<dev, scalar_t> tgt_featmap,
                        const torch::Tensor &src_points,
                        const torch::Tensor &src_feats,
                        const torch::Tensor &src_mask,
                        const torch::Tensor &kcam, const torch::Tensor &rt_cam,
                        torch::Tensor JtJ_partial, torch::Tensor Jtr_partial,
                        torch::Tensor squared_residual,
                        AtomicInt<dev> match_count)
      : tgt(tgt),
        tgt_featmap(tgt_featmap),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jtr_partial(Accessor<dev, scalar_t, 2>::Get(Jtr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int ri) {
    JacobianType jacobian(JtJ_partial[ri], Jtr_partial[ri]);

    squared_residual[ri] = 0.0;

    if (src_mask[ri] == 0) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));

    scalar_t u, v;
    kcam.Project(Tsrc_point, u, v);

    const int ui = int(round(u));
    const int vi = int(round(v));

    if (tgt.empty(vi, ui)) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;

    ++match_count;


    const BilinearInterp<dev, scalar_t> interp(tgt_featmap.GetBilinear(u, v));
    const BilinearInterpGrad<dev, scalar_t> dx_interp(
        tgt_featmap.GetBilinearGrad(u, v, 0.05));

    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      const scalar_t feat_residual = (interp.Get(0) - src_feats[0][ri]);
      scalar_t du, dv;

      dx_interp.Get(channel, du, dv);

      scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
      kcam.Dx_Projection(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);

      const Vector<scalar_t, 3> gradk(du * j00_proj, dv * j11_proj,
                                      du * j02_proj + dv * j12_proj);

      jacobian.Compute(Tsrc_point, gradk, feat_residual);
      squared_residual[ri] += feat_residual * feat_residual;
    }
  }
};
}  // namespace

int ICPJacobian::EstimateFeature(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_feats, const torch::Tensor &tgt_mask,
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &kcam,
    const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
    torch::Tensor Jr_partial, torch::Tensor squared_residual) {
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
  FTB_CHECK_DEVICE(reference_dev, squared_residual);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", ([&] {
          PointGrid<kCUDA, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, SE3Jacobian<kCUDA, scalar_t> >
              kernel(tgt, FeatureMap<kCUDA, scalar_t>(tgt_feats), src_points,
                     src_feats, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
                     squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeature", [&] {
          PointGrid<kCPU, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCPU> match_count;

          FeatureJacobianKernel<kCPU, scalar_t, SE3Jacobian<kCPU, scalar_t> >
              kernel(tgt, FeatureMap<kCPU, scalar_t>(tgt_feats), src_points,
                     src_feats, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
                     squared_residual, match_count.get());
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
    const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
    torch::Tensor Jr_partial, torch::Tensor squared_residual) {
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
  FTB_CHECK_DEVICE(reference_dev, squared_residual);

  int num_matches;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeatureSO3", ([&] {
          PointGrid<kCUDA, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCUDA> match_count;

          FeatureJacobianKernel<kCUDA, scalar_t, SO3Jacobian<kCUDA, scalar_t> >
              kernel(tgt, FeatureMap<kCUDA, scalar_t>(tgt_feats), src_points,
                     src_feats, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
                     squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeatureSO3", [&] {
          PointGrid<kCPU, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCPU> match_count;

          FeatureJacobianKernel<kCPU, scalar_t, SO3Jacobian<kCPU, scalar_t> >
              kernel(tgt, FeatureMap<kCPU, scalar_t>(tgt_feats), src_points,
                     src_feats, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
                     squared_residual, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

}  // namespace fiontb
