#include "icpodometry.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "error.hpp"
#include "feature_map.hpp"
#include "kernel.hpp"
#include "pointgrid.hpp"

namespace fiontb {

namespace {
template <Device dev, typename scalar_t>
class PointGrid : public BasePointGrid<dev> {
 public:
  const typename Accessor<dev, scalar_t, 3>::T points;
  const typename Accessor<dev, scalar_t, 3>::T normals;

  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
            const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, scalar_t, 3>::Get(points)),
        normals(Accessor<dev, scalar_t, 3>::Get(normals)) {}
};

template <Device dev, typename scalar_t>
FTB_DEVICE_HOST inline scalar_t EuclideanDistance(
    const BilinearInterp<dev, scalar_t> f1,
    const typename Accessor<dev, scalar_t, 2>::T f2, int f2_index) {
  scalar_t dist = scalar_t(0);
  for (int channel = 0; channel < f2.size(0); ++channel) {
    const scalar_t diff = f1.Get(channel) - f2[channel][f2_index];
    dist += diff * diff;
  }

  return sqrt(dist);
}

template <typename scalar_t>
FTB_DEVICE_HOST inline scalar_t Df1_EuclideanDistance(
    scalar_t f1_nth_val, scalar_t f2_nth_val, scalar_t inv_forward_result) {
  if (inv_forward_result > 0)
    return (f2_nth_val - f1_nth_val) * inv_forward_result;
  else
    return 0;
}

template <Device dev, typename scalar_t>
struct FeatureSO3JacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;
  const KCamera<dev, scalar_t> kcam;
  const RTCamera<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;
  FeatureSO3JacobianKernel(
      const PointGrid<dev, scalar_t> tgt, FeatureMap<dev, scalar_t> tgt_featmap,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const KCamera<dev, scalar_t> kcam,
      const RTCamera<dev, scalar_t> rt_cam, torch::Tensor JtJ_partial,
      torch::Tensor Jr_partial, torch::Tensor squared_residual,
      AtomicInt<dev> match_count)
      : tgt(tgt),
        tgt_featmap(tgt_featmap),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jr_partial(Accessor<dev, scalar_t, 2>::Get(Jr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

  FTB_DEVICE_HOST void operator()(int ri) {
    for (int k = 0; k < 3; k++) Jr_partial[ri][k] = 0;

    for (int krow = 0; krow < 3; krow++) {
      for (int kcol = 0; kcol < 3; kcol++) {
        JtJ_partial[ri][krow][kcol] = 0;
      }
    }

    squared_residual[ri] = 0;

    if (src_mask[ri] == 0) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));

    scalar_t u, v;
    kcam.Project(Tsrc_point, u, v);

    const int ui = int(round(u));
    const int vi = int(round(v));
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;
    if (tgt.empty(vi, ui)) return;

    ++match_count;

    scalar_t jacobian[6], residual;

    BilinearInterp<dev, scalar_t> interp = tgt_featmap.GetBilinear(u, v);

    const scalar_t feat_residual = EuclideanDistance(interp, src_feats, ri);

    const scalar_t inv_feat_residual =
        (feat_residual > 0) ? scalar_t(1) / feat_residual : -1;

    const BilinearInterpGrad<dev, scalar_t> dx_interp(
        tgt_featmap.GetBilinearGrad(u, v));

    scalar_t d_euc_u = 0;
    scalar_t d_euc_v = 0;
    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      const scalar_t df1_dist = Df1_EuclideanDistance(
          interp.Get(channel), src_feats[channel][ri], inv_feat_residual);

      scalar_t du, dv;
      dx_interp.Get(channel, du, dv);

      d_euc_u += df1_dist * du;
      d_euc_v += df1_dist * dv;
    }

    d_euc_v *= -1;

    scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
    kcam.Dx_Projection(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);

    Vector<scalar_t, 3> gradk(d_euc_u * j00_proj, d_euc_v * j11_proj,
                              d_euc_u * j02_proj + d_euc_v * j12_proj);
    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(gradk);
    jacobian[0] = rot_twist[0];
    jacobian[1] = rot_twist[1];
    jacobian[2] = rot_twist[2];

    residual = feat_residual;

    squared_residual[ri] = residual * residual;

    for (int k = 0; k < 3; ++k) {
      Jr_partial[ri][k] = jacobian[k] * residual;
    }

    for (int krow = 0; krow < 3; ++krow) {
      for (int kcol = krow; kcol < 3; ++kcol) {
        const scalar_t v = jacobian[kcol] * jacobian[krow];
        JtJ_partial[ri][krow][kcol] = v;
        JtJ_partial[ri][kcol][krow] = v;
      }
    }
  }
};
}  // namespace

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

  int num_matches = 0;
  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeatureSO3", ([&] {
          PointGrid<kCUDA, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCUDA> match_count;

          FeatureSO3JacobianKernel<kCUDA, scalar_t> kernel(
              tgt, FeatureMap<kCUDA, scalar_t>(tgt_feats), src_points,
              src_feats, src_mask, KCamera<kCUDA, scalar_t>(kcam),
              RTCamera<kCUDA, scalar_t>(rt_cam), JtJ_partial, Jr_partial,
              squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));
          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateFeatureSO3", [&] {
          PointGrid<kCPU, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCPU> match_count;

          FeatureSO3JacobianKernel<kCPU, scalar_t> kernel(
              tgt, FeatureMap<kCPU, scalar_t>(tgt_feats), src_points, src_feats,
              src_mask, KCamera<kCPU, scalar_t>(kcam),
              RTCamera<kCPU, scalar_t>(rt_cam), JtJ_partial, Jr_partial,
              squared_residual, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));
          num_matches = match_count;
        });
  }
  return num_matches;
}

namespace {
template <Device dev, typename scalar_t>
struct GeometricJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, bool, 1>::T src_mask;
  const KCamera<dev, scalar_t> kcam;
  const RTCamera<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  GeometricJacobianKernel(PointGrid<dev, scalar_t> tgt,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_mask,
                          const KCamera<dev, scalar_t> kcam,
                          const RTCamera<dev, scalar_t> rt_cam,
                          torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
                          torch::Tensor squared_residual,
                          AtomicInt<dev> match_count)
      : tgt(tgt),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jr_partial(Accessor<dev, scalar_t, 2>::Get(Jr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

  FTB_DEVICE_HOST void operator()(int ri) {
#pragma unroll
    for (int k = 0; k < 6; k++) {
      Jr_partial[ri][k] = 0;
    }

#pragma unroll
    for (int krow = 0; krow < 6; krow++) {
#pragma unroll
      for (int kcol = 0; kcol < 6; kcol++) {
        JtJ_partial[ri][krow][kcol] = 0;
      }
    }

    squared_residual[ri] = 0;

    if (src_mask[ri] == 0) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));
    Eigen::Vector2i src_uv = kcam.Project(Tsrc_point);
    if (src_uv[0] < 0 || src_uv[0] >= width || src_uv[1] < 0 ||
        src_uv[1] >= height)
      return;

    if (tgt.empty(src_uv[1], src_uv[0])) return;

    ++match_count;

    const Vector<scalar_t, 3> tgt_point(
        to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]));
    const Vector<scalar_t, 3> tgt_normal(
        to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]));

    scalar_t jacobian[6];
    jacobian[0] = tgt_normal[0];
    jacobian[1] = tgt_normal[1];
    jacobian[2] = tgt_normal[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(tgt_normal);
    jacobian[3] = rot_twist[0];
    jacobian[4] = rot_twist[1];
    jacobian[5] = rot_twist[2];

    const scalar_t residual = (tgt_point - Tsrc_point).dot(tgt_normal);
    squared_residual[ri] = residual * residual;

#pragma unroll
    for (int k = 0; k < 6; ++k) {
      Jr_partial[ri][k] = jacobian[k] * residual;
    }
#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[ri][krow][kcol] = jacobian[kcol] * jacobian[krow];
      }
    }
  }
};

}  // namespace

int ICPJacobian::EstimateGeometric(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_mask, const torch::Tensor &src_points,
    const torch::Tensor &src_mask, const torch::Tensor &kcam,
    const torch::Tensor &rt_cam, torch::Tensor JtJ_partial,
    torch::Tensor Jr_partial, torch::Tensor squared_residual) {
  const auto reference_dev = src_points.device();

  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_mask);
  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);

  FTB_CHECK_DEVICE(reference_dev, JtJ_partial);
  FTB_CHECK_DEVICE(reference_dev, Jr_partial);
  FTB_CHECK_DEVICE(reference_dev, squared_residual);

  int num_matches;

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          ScopedAtomicInt<kCUDA> match_count;
          GeometricJacobianKernel<kCUDA, scalar_t> kernel(
              PointGrid<kCUDA, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
              squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          ScopedAtomicInt<kCPU> match_count;
          GeometricJacobianKernel<kCPU, scalar_t> kernel(
              PointGrid<kCPU, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, JtJ_partial, Jr_partial,
              squared_residual, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

namespace {

template <Device dev, typename scalar_t>
struct HybridJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const KCamera<dev, scalar_t> kcam;
  const RTCamera<dev, scalar_t> rt_cam;

  scalar_t geom_weight, feat_weight;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  HybridJacobianKernel(
      const PointGrid<dev, scalar_t> tgt, FeatureMap<dev, scalar_t> tgt_featmap,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const KCamera<dev, scalar_t> kcam,
      const RTCamera<dev, scalar_t> rt_cam, scalar_t geom_weight,
      scalar_t feat_weight, torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
      torch::Tensor squared_residual, AtomicInt<dev> match_count)
      : tgt(tgt),
        tgt_featmap(tgt_featmap),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        geom_weight(geom_weight),
        feat_weight(feat_weight),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jr_partial(Accessor<dev, scalar_t, 2>::Get(Jr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

  FTB_DEVICE_HOST void ComputeGeometricTerm(
      const Vector<scalar_t, 3> &Tsrc_point, int ui, int vi,
      scalar_t out_jacobian[6], scalar_t &out_residual) {
    const Vector<scalar_t, 3> tgt_point(to_vec3<scalar_t>(tgt.points[vi][ui]));
    const Vector<scalar_t, 3> tgt_normal(
        to_vec3<scalar_t>(tgt.normals[vi][ui]));

    out_jacobian[0] = tgt_normal[0];
    out_jacobian[1] = tgt_normal[1];
    out_jacobian[2] = tgt_normal[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(tgt_normal);
    out_jacobian[3] = rot_twist[0];
    out_jacobian[4] = rot_twist[1];
    out_jacobian[5] = rot_twist[2];

    out_residual = tgt_normal.dot(tgt_point - Tsrc_point);
  }

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void ComputeFeatTerm(int ri,
                                       const Vector<scalar_t, 3> &Tsrc_point,
                                       scalar_t u, scalar_t v,
                                       scalar_t out_jacobian[6],
                                       scalar_t &out_residual) {
    BilinearInterp<dev, scalar_t> interp = tgt_featmap.GetBilinear(u, v);

    const scalar_t feat_residual = EuclideanDistance(interp, src_feats, ri);

    const scalar_t inv_feat_residual =
        (feat_residual > 0) ? scalar_t(1) / feat_residual : -1;

    const BilinearInterpGrad<dev, scalar_t> dx_interp(
        tgt_featmap.GetBilinearGrad(u, v));

    scalar_t d_euc_u = 0;
    scalar_t d_euc_v = 0;
    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      const scalar_t df1_dist = Df1_EuclideanDistance(
          interp.Get(channel), src_feats[channel][ri], inv_feat_residual);

      scalar_t du, dv;
      dx_interp.Get(channel, du, dv);

      d_euc_u += df1_dist * du;
      d_euc_v += df1_dist * dv;
    }

    d_euc_v *= -1;

    scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
    kcam.Dx_Projection(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);

    Vector<scalar_t, 3> gradk(d_euc_u * j00_proj, d_euc_v * j11_proj,
                              d_euc_u * j02_proj + d_euc_v * j12_proj);
    out_jacobian[0] = gradk[0];
    out_jacobian[1] = gradk[1];
    out_jacobian[2] = gradk[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(gradk);
    out_jacobian[3] = rot_twist[0];
    out_jacobian[4] = rot_twist[1];
    out_jacobian[5] = rot_twist[2];

    out_residual = feat_residual;
  }

  FTB_DEVICE_HOST void operator()(int ri) {
    for (int k = 0; k < 6; k++) Jr_partial[ri][k] = 0;

    for (int krow = 0; krow < 6; krow++) {
      for (int kcol = 0; kcol < 6; kcol++) {
        JtJ_partial[ri][krow][kcol] = 0;
      }
    }

    squared_residual[ri] = 0.0;

    if (ri == 484) {
      squared_residual[ri] = 0.0001;
    }

    if (src_mask[ri] == 0) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));

    scalar_t u, v;
    kcam.Project(Tsrc_point, u, v);

    const int ui = int(round(u));
    const int vi = int(round(v));
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;
    if (tgt.empty(vi, ui)) return;

    const float zdiff = abs(Tsrc_point[2] - tgt.points[vi][ui][2]);
    if (zdiff > 0.5) return;

    ++match_count;

    scalar_t feat_jacobian[6], feat_residual;
    ComputeFeatTerm(ri, Tsrc_point, u, v, feat_jacobian, feat_residual);

    scalar_t geom_jacobian[6], geom_residual;
    ComputeGeometricTerm(Tsrc_point, ui, vi, geom_jacobian, geom_residual);

    scalar_t geom_JtJ_partial[6][6];
    scalar_t geom_Jtr_partial[6];

    scalar_t feat_JtJ_partial[6][6];
    scalar_t feat_Jtr_partial[6];

    for (int k = 0; k < 6; ++k) {
      geom_Jtr_partial[k] = geom_jacobian[k] * geom_residual;
      feat_Jtr_partial[k] = feat_jacobian[k] * feat_residual;
    }

    for (int krow = 0; krow < 6; ++krow) {
      for (int kcol = krow; kcol < 6; ++kcol) {
        const scalar_t geom_v = geom_jacobian[kcol] * geom_jacobian[krow];
        geom_JtJ_partial[krow][kcol] = geom_v;
        geom_JtJ_partial[kcol][krow] = geom_v;

        const scalar_t feat_v = feat_jacobian[kcol] * feat_jacobian[krow];
        feat_JtJ_partial[krow][kcol] = feat_v;
        feat_JtJ_partial[kcol][krow] = feat_v;
      }
    }

    const scalar_t residual =
        geom_residual * geom_weight + feat_residual * feat_weight;

    squared_residual[ri] = residual * residual;

    for (int k = 0; k < 6; ++k) {
      Jr_partial[ri][k] =
          geom_Jtr_partial[k] * geom_weight + feat_Jtr_partial[k] * feat_weight;
    }

    for (int krow = 0; krow < 6; ++krow) {
      for (int kcol = krow; kcol < 6; ++kcol) {
        const scalar_t v = geom_JtJ_partial[krow][kcol] * geom_weight +
                           feat_JtJ_partial[krow][kcol] * feat_weight;

        JtJ_partial[ri][krow][kcol] = v;
        JtJ_partial[ri][kcol][krow] = v;
      }
    }
  }
};

}  // namespace

int ICPJacobian::EstimateHybrid(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_feats, const torch::Tensor &tgt_mask,
    const torch::Tensor &src_points, const torch::Tensor &src_feats,
    const torch::Tensor &src_mask, const torch::Tensor &kcam,
    const torch::Tensor &rt_cam, float geom_weight, float feat_weight,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residual) {
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
        src_points.scalar_type(), "EstimateHybrid", ([&] {
          PointGrid<kCUDA, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          ScopedAtomicInt<kCUDA> match_count;

          HybridJacobianKernel<kCUDA, scalar_t> kernel(
              tgt, FeatureMap<kCUDA, scalar_t>(tgt_feats), src_points,
              src_feats, src_mask, KCamera<kCUDA, scalar_t>(kcam),
              RTCamera<kCUDA, scalar_t>(rt_cam), geom_weight, feat_weight,
              JtJ_partial, Jr_partial, squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(src_points.scalar_type(), "EstimateHybrid", [&] {
      PointGrid<kCPU, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
      ScopedAtomicInt<kCPU> match_count;

      HybridJacobianKernel<kCPU, scalar_t> kernel(
          tgt, FeatureMap<kCPU, scalar_t>(tgt_feats), src_points, src_feats,
          src_mask, KCamera<kCPU, scalar_t>(kcam),
          RTCamera<kCPU, scalar_t>(rt_cam), geom_weight, feat_weight,
          JtJ_partial, Jr_partial, squared_residual, match_count.get());
      Launch1DKernelCPU(kernel, src_points.size(0));

      num_matches = match_count;
    });
  }
  return num_matches;
}

void ICPJacobian::RegisterPybind(pybind11::module &m) {
  py::class_<ICPJacobian>(m, "ICPJacobian")
      .def_static("estimate_geometric", &ICPJacobian::EstimateGeometric)
      .def_static("estimate_hybrid", &ICPJacobian::EstimateHybrid)
      .def_static("estimate_feature_so3", &ICPJacobian::EstimateFeatureSO3);
}
}  // namespace fiontb
