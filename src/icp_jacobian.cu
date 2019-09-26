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
    return (f1_nth_val - f2_nth_val) * inv_forward_result;
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

	scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
    kcam.Dx_Projection(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);
	
    Eigen::Matrix<scalar_t, 1, 3> pgrad;
    pgrad << d_euc_u * j00_proj, d_euc_v * j11_proj,
        d_euc_u * j02_proj + d_euc_v * j12_proj;
	
    Eigen::Matrix<scalar_t, 3, 3> K;
    K << kcam.matrix[0][0], kcam.matrix[0][1], kcam.matrix[0][2],
        kcam.matrix[1][0], kcam.matrix[1][1], kcam.matrix[1][2],
        kcam.matrix[2][0], kcam.matrix[2][1], kcam.matrix[2][2];

    Eigen::Matrix<scalar_t, 3, 6> J;
    // clang-format off
	J <<
	  1, 0, 0, 0, Tsrc_point[2], -Tsrc_point[1],
	  0, 1, 0, -Tsrc_point[2], 0, Tsrc_point[0],
	  0, 0, 1, Tsrc_point[1], -Tsrc_point[0], 0;
    // clang-format on

    //J = K * J;
    Eigen::Matrix<scalar_t, 1, 6> jacobian = pgrad * J;

    for (int k = 0; k < 6; ++k) out_jacobian[k] = jacobian(0, k);

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
    scalar_t geom_jacobian[6], geom_residual;

    if (ri == 228479) {
      geom_residual = 0;
    }
    ComputeFeatTerm(ri, Tsrc_point, u, v, feat_jacobian, feat_residual);
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
