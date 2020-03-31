#include "icp_jacobian.hpp"

#include "accessor.hpp"
#include "atomic_int.hpp"
#include "camera.hpp"
#include "correspondence.hpp"
#include "error.hpp"
#include "icp_jacobian_common.hpp"
#include "kernel.hpp"
#include "merge_map.hpp"

namespace slamtb {

namespace {
template <Device dev, typename scalar_t>
struct GeometricJacobianKernel {
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const RigidTransform<scalar_t> rt_cam;

  const typename Accessor<dev, float, 3>::T target_points;
  const typename Accessor<dev, float, 3>::T target_normals;
  const KCamera<dev, scalar_t> kcam;

  MergeMapAccessor<dev> merge_map;

  typename Accessor<dev, scalar_t, 3>::T JtJ_partial;
  typename Accessor<dev, scalar_t, 2>::T Jtr_partial;
  typename Accessor<dev, scalar_t, 1>::T squared_residual;

  AtomicInt<dev> match_count;

  GeometricJacobianKernel(
      const torch::Tensor &src_points, const torch::Tensor &src_mask,
      const torch::Tensor &rt_cam, const torch::Tensor &target_points,
      const torch::Tensor &target_normals, const torch::Tensor &kcam,
      const torch::Tensor &merge_map, torch::Tensor JtJ_partial,
      torch::Tensor Jtr_partial, torch::Tensor squared_residual,
      AtomicInt<dev> match_count)
      : src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        rt_cam(rt_cam),
        target_points(Accessor<dev, float, 3>::Get(target_points)),
        target_normals(Accessor<dev, float, 3>::Get(target_normals)),
        kcam(kcam),
        merge_map(merge_map),
        JtJ_partial(Accessor<dev, scalar_t, 3>::Get(JtJ_partial)),
        Jtr_partial(Accessor<dev, scalar_t, 2>::Get(Jtr_partial)),
        squared_residual(Accessor<dev, scalar_t, 1>::Get(squared_residual)),
        match_count(match_count) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int source_idx) {
    SE3ICPJacobian<dev, scalar_t> jacobian(JtJ_partial[source_idx],
                                           Jtr_partial[source_idx]);
    squared_residual[source_idx] = 0;
    if (!src_mask[source_idx]) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[source_idx]));

    int ui, vi;
    kcam.Projecti(Tsrc_point, ui, vi);
    
    if (ui < 0 || ui >= target_points.size(1) || vi < 0 ||
        vi >= target_points.size(0))
      return;

    const int32_t target_index = merge_map(vi, ui);
    if (target_index != source_idx) {
      return;
    }

    const Vector<scalar_t, 3> tgt_point(
        to_vec3<scalar_t>(target_points[vi][ui]));
    const Vector<scalar_t, 3> tgt_normal(
        to_vec3<scalar_t>(target_normals[vi][ui]));

    ++match_count;

    const scalar_t residual = (tgt_point - Tsrc_point).dot(tgt_normal);
    jacobian.Compute(Tsrc_point, tgt_normal, residual);
    squared_residual[source_idx] = residual * residual;
  }
};

}  // namespace

int ICPJacobian::EstimateGeometric(
    const torch::Tensor &src_points, const torch::Tensor &src_mask,
    const torch::Tensor &rt_cam, const torch::Tensor &tgt_points,
    const torch::Tensor &tgt_normals, const torch::Tensor &kcam,
    const torch::Tensor &merge_map, torch::Tensor JtJ_partial,
    torch::Tensor Jr_partial, torch::Tensor squared_residual) {
  const auto reference_dev = src_points.device();

  FTB_CHECK_DEVICE(reference_dev, tgt_points);
  FTB_CHECK_DEVICE(reference_dev, tgt_normals);

  FTB_CHECK_DEVICE(reference_dev, src_mask);
  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, merge_map);

  FTB_CHECK_DEVICE(reference_dev, JtJ_partial);
  FTB_CHECK_DEVICE(reference_dev, Jr_partial);
  FTB_CHECK_DEVICE(reference_dev, squared_residual);

  int num_matches;

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          ScopedAtomicInt<kCUDA> match_count;

          GeometricJacobianKernel<kCUDA, scalar_t> kernel(
              src_points, src_mask, rt_cam, tgt_points, tgt_normals, kcam,
              merge_map, JtJ_partial, Jr_partial, squared_residual,
              match_count.get());

          Launch1DKernelCUDA(kernel, src_points.size(0));

          num_matches = match_count;
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          ScopedAtomicInt<kCPU> match_count;

          GeometricJacobianKernel<kCPU, scalar_t> kernel(
              src_points, src_mask, rt_cam, tgt_points, tgt_normals, kcam,
              merge_map, JtJ_partial, Jr_partial, squared_residual,
              match_count.get());
          Launch1DKernelCPU(kernel, src_points.size(0));

          num_matches = match_count;
        });
  }
  return num_matches;
}

}  // namespace slamtb
