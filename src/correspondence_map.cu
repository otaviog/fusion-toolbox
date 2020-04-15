#include "matching.hpp"

#include "accessor.hpp"
#include "correspondence.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "merge_map.hpp"

namespace slamtb {
namespace {
template <Device dev, typename scalar_t>
struct CorrespondenceMapKernel {
  const typename Accessor<dev, scalar_t, 2>::T source_points;
  const typename Accessor<dev, scalar_t, 2>::T source_normals;
  const typename Accessor<dev, bool, 1>::T source_mask;
  const RigidTransform<scalar_t> rt_cam;

  RobustCorrespondence<dev, scalar_t> corresp;
  MergeMap<dev> merge_map;

  CorrespondenceMapKernel(const torch::Tensor &source_points,
                          const torch::Tensor &source_normals,
                          const torch::Tensor &source_mask,
                          const torch::Tensor &rt_cam,
                          const torch::Tensor &target_points,
                          const torch::Tensor &target_normals,
                          const torch::Tensor &target_mask,
                          const torch::Tensor &kcam, torch::Tensor merge_map,
                          double distance_thresh, double normal_angle_thesh)
      : source_points(Accessor<dev, scalar_t, 2>::Get(source_points)),
        source_normals(Accessor<dev, scalar_t, 2>::Get(source_normals)),
        source_mask(Accessor<dev, bool, 1>::Get(source_mask)),
        rt_cam(rt_cam),
        corresp(target_points, target_normals, target_mask, kcam,
                distance_thresh, normal_angle_thesh),
        merge_map(merge_map) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(int idx) {
	if (source_mask.size(0) == source_points.size(0)
		&& !source_mask[idx]) return;

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(source_points[idx]));
    const Vector<scalar_t, 3> Tsrc_normal =
        rt_cam.TransformNormal(to_vec3<scalar_t>(source_normals[idx]));

    scalar_t u, v;
    if (!corresp.Match(Tsrc_point, Tsrc_normal, u, v)) {
      return;
    }

    const int ui = int(round(u));
    const int vi = int(round(v));

    const Vector<scalar_t, 3> target(
        to_vec3<scalar_t>(corresp.tgt.points[vi][ui]));
    const scalar_t dist = (target - Tsrc_point).norm();

    merge_map.Set(vi, ui, float(dist), idx);
  }
};
}  // namespace
void CorrespondenceMap::ComputeCorrespondenceMap(
    const torch::Tensor &source_points, const torch::Tensor &source_normals,
    const torch::Tensor &source_mask, const torch::Tensor &rt_cam,
    const torch::Tensor &target_points, const torch::Tensor &target_normals,
    const torch::Tensor &target_mask, const torch::Tensor &kcam,
    torch::Tensor merge_map, double distance_thresh,
    double normal_angle_thesh) {
  const auto reference_dev = source_points.device();

  FTB_CHECK_DEVICE(reference_dev, source_normals);
  FTB_CHECK_DEVICE(reference_dev, source_mask);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);

  FTB_CHECK_DEVICE(reference_dev, target_points);
  FTB_CHECK_DEVICE(reference_dev, target_normals);
  FTB_CHECK_DEVICE(reference_dev, target_mask);

  FTB_CHECK_DEVICE(reference_dev, kcam);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        source_points.scalar_type(), "ComputeCorrespondenceMap", ([&] {
          CorrespondenceMapKernel<kCUDA, scalar_t> kernel(
              source_points, source_normals, source_mask, rt_cam, target_points,
              target_normals, target_mask, kcam, merge_map, distance_thresh,
              normal_angle_thesh);
          Launch1DKernelCUDA(kernel, source_points.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        source_points.scalar_type(), "ComputeCorrespondenceMap", ([&] {
          CorrespondenceMapKernel<kCPU, scalar_t> kernel(
              source_points, source_normals, source_mask, rt_cam, target_points,
              target_normals, target_mask, kcam, merge_map, distance_thresh,
              normal_angle_thesh);
          Launch1DKernelCPU(kernel, source_points.size(0));
        }));
  }
}

}  // namespace slamtb
