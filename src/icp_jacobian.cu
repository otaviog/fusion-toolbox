#include "icpodometry.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "error.hpp"

#include "icp_jacobian_common.hpp"
#include "kernel.hpp"

#include "correspondence.hpp"
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
  typename Accessor<dev, scalar_t, 2>::T Jr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  GeometricJacobianKernel(Correspondence corresp,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_normals,
                          const torch::Tensor &src_mask,
                          const torch::Tensor &rt_cam,
                          torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
                          torch::Tensor squared_residual,
                          AtomicInt<dev> match_count)
      : corresp(corresp),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_normals(Accessor<dev, scalar_t, 2>::Get(src_normals)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
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
    const torch::Tensor &src_normals, const torch::Tensor &src_mask,
    const torch::Tensor &kcam, const torch::Tensor &rt_cam,
    torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
    torch::Tensor squared_residual) {
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

void ICPJacobian::RegisterPybind(pybind11::module &m) {
  py::class_<ICPJacobian>(m, "ICPJacobian")
      .def_static("estimate_geometric", &ICPJacobian::EstimateGeometric)
      .def_static("estimate_feature", &ICPJacobian::EstimateFeature)
      .def_static("estimate_feature_so3", &ICPJacobian::EstimateFeatureSO3);
}
}  // namespace fiontb
