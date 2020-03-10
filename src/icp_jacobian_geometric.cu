#include "icp_jacobian.hpp"

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "correspondence.hpp"
#include "error.hpp"
#include "icp_jacobian_common.hpp"
#include "kernel.hpp"

namespace fiontb {

namespace {
template <Device dev, typename scalar_t, typename Correspondence>
struct GeometricJacobianKernel {
  const Correspondence corresp;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 2>::T src_normals;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const RTCamera<scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jtr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  GeometricJacobianKernel(Correspondence corresp,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_normals,
                          const torch::Tensor &src_mask,
                          const torch::Tensor &rt_cam,
                          torch::Tensor JtJ_partial, torch::Tensor Jtr_partial,
                          torch::Tensor squared_residual,
                          AtomicInt<dev> match_count)
      : corresp(corresp),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_normals(Accessor<dev, scalar_t, 2>::Get(src_normals)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        rt_cam(rt_cam),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jtr_partial(Accessor<dev, scalar_t, 2>::Get(Jtr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

  FTB_DEVICE_HOST void operator()(int ri) {
    SE3ICPJacobian<dev, scalar_t> jacobian(JtJ_partial[ri], Jtr_partial[ri]);
    squared_residual[ri] = 0;
    if (!src_mask[ri]) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));
    const Vector<scalar_t, 3> Tsrc_normal =
        rt_cam.TransformNormal(to_vec3<scalar_t>(src_normals[ri]));

    Vector<scalar_t, 3> tgt_point;
    Vector<scalar_t, 3> tgt_normal;
    if (!corresp.Match(Tsrc_point, Tsrc_normal, tgt_point, tgt_normal)) {
      return;
    }

    ++match_count;

    const scalar_t residual = (tgt_point - Tsrc_point).dot(tgt_normal);
    jacobian.Compute(Tsrc_point, tgt_normal, residual);
    squared_residual[ri] = residual * residual;
  }
};

}  // namespace

int ICPJacobian::EstimateGeometric(
    const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
    const torch::Tensor &tgt_mask, const torch::Tensor &src_points,
    const torch::Tensor &src_normals, const torch::Tensor &src_mask,
    const torch::Tensor &kcam, const torch::Tensor &rt_cam,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residual) {
  const auto reference_dev = src_points.device();

  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);
  FTB_CHECK_DEVICE(reference_dev, tgt_mask);

  FTB_CHECK_DEVICE(reference_dev, src_mask);
  FTB_CHECK_DEVICE(reference_dev, src_normals);
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

          typedef RobustCorrespondence<kCUDA, scalar_t> Correspondence;

          GeometricJacobianKernel<kCUDA, scalar_t, Correspondence> kernel(
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam),
              src_points, src_normals, src_mask, rt_cam, JtJ_partial,
              Jr_partial, squared_residual, match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          ScopedAtomicInt<kCPU> match_count;

          typedef RobustCorrespondence<kCPU, scalar_t> Correspondence;

          GeometricJacobianKernel<kCPU, scalar_t, Correspondence> kernel(
              Correspondence(tgt_points, tgt_normals, tgt_mask, kcam),
              src_points, src_normals, src_mask, rt_cam, JtJ_partial,
              Jr_partial, squared_residual, match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

}  // namespace fiontb
