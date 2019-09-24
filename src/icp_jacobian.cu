#include "icpodometry.hpp"

#include "accessor.hpp"
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
struct GeometricJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, bool, 1>::T src_mask;
  KCamera<dev, scalar_t> kcam;
  RTCamera<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 2>::T jacobian;
  typename Accessor<dev, scalar_t, 1>::T residual;

  GeometricJacobianKernel(PointGrid<dev, scalar_t> tgt,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_mask,
                          KCamera<dev, scalar_t> kcam,
                          RTCamera<dev, scalar_t> rt_cam,
                          torch::Tensor jacobian, torch::Tensor residual)
      : tgt(tgt),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(Accessor<dev, scalar_t, 2>::Get(jacobian)),
        residual(Accessor<dev, scalar_t, 1>::Get(residual)) {}

  FTB_DEVICE_HOST void operator()(int ri) {
    jacobian[ri][0] = 0.0f;
    jacobian[ri][1] = 0.0f;
    jacobian[ri][2] = 0.0f;
    jacobian[ri][3] = 0.0f;
    jacobian[ri][4] = 0.0f;
    jacobian[ri][5] = 0.0f;
    residual[ri] = 0.0f;

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
    const Vector<scalar_t, 3> tgt_point(
        to_vec3<scalar_t>(tgt.points[src_uv[1]][src_uv[0]]));
    const Vector<scalar_t, 3> tgt_normal(
        to_vec3<scalar_t>(tgt.normals[src_uv[1]][src_uv[0]]));

    jacobian[ri][0] = tgt_normal[0];
    jacobian[ri][1] = tgt_normal[1];
    jacobian[ri][2] = tgt_normal[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(tgt_normal);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = (tgt_point - Tsrc_point).dot(tgt_normal);
  }
};

}  // namespace

void ICPJacobian::EstimateGeometric(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_mask, const torch::Tensor kcam,
    const torch::Tensor rt_cam, torch::Tensor jacobian,
    torch::Tensor residual) {
  const auto reference_dev = src_points.device();

  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_mask);
  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);

  FTB_CHECK_DEVICE(reference_dev, jacobian);
  FTB_CHECK_DEVICE(reference_dev, residual);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          GeometricJacobianKernel<kCUDA, scalar_t> kernel(
              PointGrid<kCUDA, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, jacobian, residual);
          Launch1DKernelCUDA(kernel, src_points.size(0));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          GeometricJacobianKernel<kCPU, scalar_t> kernel(
              PointGrid<kCPU, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, jacobian, residual);
          Launch1DKernelCPU(kernel, src_points.size(0));
        });
  }
}

namespace {
template <Device dev, typename scalar_t>
FTB_DEVICE_HOST scalar_t EuclideanDistance(
    const ChannelInterpolator<dev, scalar_t> lhs,
    const typename Accessor<dev, scalar_t, 2>::T rhs, int rhs_index) {
  scalar_t dist = scalar_t(0);
  for (int idx = 0; idx < rhs.size(0); ++idx) {
    dist += pow(lhs.Get(idx) - rhs[idx][rhs_index], 2);
  }

  return sqrt(dist);
}

template <typename scalar_t>
FTB_DEVICE_HOST scalar_t DxEuclideanDistance(scalar_t lhs_nth_feat,
                                             scalar_t rhs_nth_feat,
                                             scalar_t forward_result) {
  if (forward_result > 0)
    return (rhs_nth_feat - lhs_nth_feat) / forward_result;
  else
    return 0;
}

template <Device dev, typename scalar_t>
struct HybridJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  FeatureMap<dev, scalar_t> tgt_featmap;

  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_feats;
  const typename Accessor<dev, bool, 1>::T src_mask;

  KCamera<dev, scalar_t> kcam;
  RTCamera<dev, scalar_t> rt_cam;

  scalar_t geom_weight, feat_weight;

  typename Accessor<dev, scalar_t, 2>::T jacobian;
  typename Accessor<dev, scalar_t, 1>::T residual;

  HybridJacobianKernel(
      const PointGrid<dev, scalar_t> tgt, FeatureMap<dev, scalar_t> tgt_featmap,
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, KCamera<dev, scalar_t> kcam,
      RTCamera<dev, scalar_t> rt_cam, scalar_t geom_weight,
      scalar_t feat_weight, torch::Tensor jacobian, torch::Tensor residual)
      : tgt(tgt),
        tgt_featmap(tgt_featmap),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_feats(Accessor<dev, scalar_t, 2>::Get(src_feats)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        geom_weight(geom_weight),
        feat_weight(feat_weight),
        jacobian(Accessor<dev, scalar_t, 2>::Get(jacobian)),
        residual(Accessor<dev, scalar_t, 1>::Get(residual)) {}

  FTB_DEVICE_HOST void ComputeGeometricTerm(
      const Vector<scalar_t, 3> &Tsrc_point, int ui, int vi,
      scalar_t out_jacobian[6], scalar_t &out_residual) {
    const Vector<scalar_t, 3> point0(to_vec3<scalar_t>(tgt.points[vi][ui]));
    const Vector<scalar_t, 3> normal0(to_vec3<scalar_t>(tgt.normals[vi][ui]));

    out_jacobian[0] = normal0[0];
    out_jacobian[1] = normal0[1];
    out_jacobian[2] = normal0[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(normal0);
    out_jacobian[3] = rot_twist[0];
    out_jacobian[4] = rot_twist[1];
    out_jacobian[5] = rot_twist[2];

    out_residual = (point0 - Tsrc_point).dot(normal0);
  }

  FTB_DEVICE_HOST void ComputeFeatTerm(int ri,
                                       const Vector<scalar_t, 3> &Tsrc_point,
                                       scalar_t u, scalar_t v,
                                       scalar_t out_jacobian[6],
                                       scalar_t &out_residual) {
    ChannelInterpolator<dev, scalar_t> channel_interp =
        tgt_featmap.InterpolateChannel(u, v);

    const scalar_t feat_residual =
        EuclideanDistance(channel_interp, src_feats, ri);

    const ChannelInterpolatorBackward<dev, scalar_t> dx_channel_interp(
        tgt_featmap.InterpolateChannelBackward(u, v));

    scalar_t d_euc_u = 0;
    scalar_t d_euc_v = 0;
    for (int channel = 0; channel < tgt_featmap.channel_size(); ++channel) {
      const scalar_t dx_dist = DxEuclideanDistance(
          channel_interp.Get(channel), src_feats[channel][ri], feat_residual);

      scalar_t du, dv;
      dx_channel_interp.Get(channel, du, dv);
      d_euc_u += dx_dist * du;
      d_euc_v += dx_dist * dv;
    }

    const Eigen::Matrix<scalar_t, 4, 1> dx_kcam(kcam.Dx_Projection(Tsrc_point));
    const Vector<scalar_t, 3> gradk(
        d_euc_u * dx_kcam[0], d_euc_v * dx_kcam[2],
        d_euc_u * dx_kcam[1] + d_euc_v * dx_kcam[3]);
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
    jacobian[ri][0] = 0.0f;
    jacobian[ri][1] = 0.0f;
    jacobian[ri][2] = 0.0f;
    jacobian[ri][3] = 0.0f;
    jacobian[ri][4] = 0.0f;
    jacobian[ri][5] = 0.0f;
    residual[ri] = 0.0f;

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

    scalar_t feat_jacobian[6], feat_residual;
    feat_jacobian[0] = feat_jacobian[1] = feat_jacobian[2] = feat_jacobian[3] =
        feat_jacobian[4] = feat_jacobian[5] = 0;
    feat_residual = 0;
    // ComputeFeatTerm(ri, Tsrc_point, u, v, feat_jacobian, feat_residual);

    scalar_t geom_jacobian[6], geom_residual;
    ComputeGeometricTerm(Tsrc_point, ui, vi, geom_jacobian, geom_residual);

#pragma unroll
    for (int k = 0; k < 6; ++k) {
      jacobian[ri][k] =
          geom_jacobian[k] * geom_weight + feat_jacobian[k] * feat_weight;
    }

    residual[ri] = geom_residual * geom_weight + feat_residual * feat_weight;
  }
};

}  // namespace

void ICPJacobian::EstimateHybrid(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_feats, const torch::Tensor tgt_mask,
    const torch::Tensor src_points, const torch::Tensor src_feats,
    const torch::Tensor src_mask, const torch::Tensor kcam,
    const torch::Tensor rt_cam, float geom_weight, float feat_weight,
    torch::Tensor jacobian, torch::Tensor residual) {
  const auto reference_dev = src_points.device();
  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_feats);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_feats);
  FTB_CHECK_DEVICE(reference_dev, src_mask);

  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, jacobian);
  FTB_CHECK_DEVICE(reference_dev, residual);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateHybrid", ([&] {
          PointGrid<kCUDA, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
          HybridJacobianKernel<kCUDA, scalar_t> kernel(
              tgt, FeatureMap<kCUDA, scalar_t>(tgt_feats), src_points,
              src_feats, src_mask, KCamera<kCUDA, scalar_t>(kcam),
              RTCamera<kCUDA, scalar_t>(rt_cam), geom_weight, feat_weight,
              jacobian, residual);
          Launch1DKernelCUDA(kernel, src_points.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(src_points.scalar_type(), "EstimateHybrid", [&] {
      PointGrid<kCPU, scalar_t> tgt(tgt_points, tgt_normals, tgt_mask);
      HybridJacobianKernel<kCPU, scalar_t> kernel(
          tgt, FeatureMap<kCPU, scalar_t>(tgt_feats), src_points, src_feats,
          src_mask, KCamera<kCPU, scalar_t>(kcam),
          RTCamera<kCPU, scalar_t>(rt_cam), geom_weight, feat_weight, jacobian,
          residual);
      Launch1DKernelCPU(kernel, src_points.size(0));
    });
  }
}
}  // namespace fiontb
