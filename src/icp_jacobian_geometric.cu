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
  STB_DEVICE_HOST void operator()(int source_idx) {
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

template <Device dev, typename scalar_t>
struct InformationMatrixKernel {
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, bool, 1>::T src_mask;

  const RigidTransform<scalar_t> rt_cam;

  const typename Accessor<dev, float, 3>::T target_points;
  const KCamera<dev, scalar_t> kcam;

  MergeMapAccessor<dev> merge_map;

  typename Accessor<dev, scalar_t, 3>::T GtG_partial;

  InformationMatrixKernel(const torch::Tensor &src_points,
                          const torch::Tensor &src_mask,
                          const torch::Tensor &rt_cam,
                          const torch::Tensor &target_points,
                          const torch::Tensor &kcam,
                          const torch::Tensor &merge_map,
                          torch::Tensor GtG_partial)
      : src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        rt_cam(rt_cam),
        target_points(Accessor<dev, float, 3>::Get(target_points)),
        kcam(kcam),
        merge_map(merge_map),
        GtG_partial(Accessor<dev, scalar_t, 3>::Get(GtG_partial)) {}

#pragma nv_exec_check_disable
  STB_DEVICE_HOST void operator()(int source_idx) {
    for (int i=0; i<6; ++i) {
      for (int j=0; j<6; ++j) {
        GtG_partial[source_idx][i][j] = 0;
      }
    }

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

    Vector<scalar_t, 6> G_r0, G_r1, G_r2;
    G_r0 << 0, tgt_point[2], -tgt_point[1], 1, 0, 0;
    G_r1 << -tgt_point[2], 0, tgt_point[0], 0, 1, 0;
    G_r2 << tgt_point[1], -tgt_point[0], 0, 0, 0, 1;

    Eigen::Matrix<scalar_t, 6, 6> GtG_local = G_r0 * G_r0.transpose();
    GtG_local += G_r1 * G_r1.transpose();
    GtG_local += G_r2 * G_r2.transpose();

    for (int i=0; i<6; ++i) {
      for (int j=0; j<6; ++j) {
        GtG_partial[source_idx][i][j] = GtG_local(i, j);
      }
    }
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

  STB_CHECK_DEVICE(reference_dev, tgt_points);
  STB_CHECK_DEVICE(reference_dev, tgt_normals);

  STB_CHECK_DEVICE(reference_dev, src_mask);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, rt_cam);
  STB_CHECK_DEVICE(reference_dev, merge_map);

  STB_CHECK_DEVICE(reference_dev, JtJ_partial);
  STB_CHECK_DEVICE(reference_dev, Jr_partial);
  STB_CHECK_DEVICE(reference_dev, squared_residual);

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

void ICPJacobian::GetInformationMatrix(
    const torch::Tensor &src_points, const torch::Tensor &src_mask,
    const torch::Tensor &rt_cam, const torch::Tensor &tgt_points,
    const torch::Tensor &kcam,
    const torch::Tensor &merge_map, torch::Tensor GtG_partial) {
  const auto reference_dev = src_points.device();

  STB_CHECK_DEVICE(reference_dev, tgt_points);
  STB_CHECK_DEVICE(reference_dev, src_mask);
  STB_CHECK_DEVICE(reference_dev, kcam);
  STB_CHECK_DEVICE(reference_dev, rt_cam);
  STB_CHECK_DEVICE(reference_dev, merge_map);

  STB_CHECK_DEVICE(reference_dev, GtG_partial);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          InformationMatrixKernel<kCUDA, scalar_t> kernel(
              src_points, src_mask, rt_cam, tgt_points, kcam,
              merge_map, GtG_partial);

          Launch1DKernelCUDA(kernel, src_points.size(0));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          InformationMatrixKernel<kCPU, scalar_t> kernel(
              src_points, src_mask, rt_cam, tgt_points, kcam,
              merge_map, GtG_partial);
          Launch1DKernelCPU(kernel, src_points.size(0));
        });
  }
}

}  // namespace slamtb
