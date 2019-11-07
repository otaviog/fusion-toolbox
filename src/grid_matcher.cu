#include "matching.hpp"

#include "camera.hpp"
#include "error.hpp"
#include "feature_map.hpp"
#include "kernel.hpp"
#include "pointgrid.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
struct ForwardKernel {
  const RobustCorrespondence<dev, scalar_t> corresp;
  const FeatureMap<dev, scalar_t> tgt_featmap;
  const typename Accessor<dev, scalar_t, 2>::T source_points;
  const typename Accessor<dev, scalar_t, 2>::T source_normals;

  typename Accessor<dev, scalar_t, 2>::T out_points;
  typename Accessor<dev, scalar_t, 2>::T out_normals;
  typename Accessor<dev, scalar_t, 2>::T out_features;
  typename Accessor<dev, bool, 1>::T match_mask;

  ForwardKernel(const torch::Tensor &target_points,
                const torch::Tensor &target_normals,
                const torch::Tensor &target_mask, const torch::Tensor kcam,
                const torch::Tensor &target_features,
                const torch::Tensor &source_points, torch::Tensor out_points,
                torch::Tensor out_normals, torch::Tensor out_features,
                torch::Tensor match_mask)
      : corresp(target_points, target_normals, target_mask, kcam),
        tgt_featmap(target_features),
        source_points(Accessor<dev, scalar_t, 2>::T(source_points)),
        source_normals(Accessor<dev, scalar_t, 2>::T(source_normals)),
        out_points(Accessor<dev, scalar_t, 2>::T(out_points)),
        out_normals(Accessor<dev, scalar_t, 2>::T(out_normals)),
        out_features(Accessor<dev, scalar_t, 2>::T(out_features)),
        match_mask(Accessor<dev, scalar_t, 2>::T(match_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    out_index[idx] = -1;

    const int height = target.points.size(0);
    const int width = target.points.size(1);

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[idx]));

    corresp.Match();
    
    int ui, vi;
    kcam.Projecti(Tsrc_point, ui, vi);

    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;
    if (target.empty(vi, ui)) return;
    auto target_point = target.points[vi][ui];
    // if (abs(Tsrc_point[2] - target_point[2]) >= 0.07) return;

    out_index[idx] = vi * width + ui;
    out_points[idx][0] = target_point[0];
    out_points[idx][1] = target_point[1];
    out_points[idx][2] = target_point[2];
  }
};

}  // namespace

void GridMatchingOp::Forward(
    const torch::Tensor &target_points, const torch::Tensor &target_normals,
    const torch::Tensor &target_features, const torch::Tensor &target_mask,
    const torch::Tensor &source_points, const torch::Tensor &kcam,
    torch::Tensor out_points, torch::Tensor out_normals,
    torch::Tensor out_features, torch::Tensor match_mask) {
  const auto reference_dev = target_points.device();
  FTB_CHECK_DEVICE(reference_dev, target_points);
  FTB_CHECK_DEVICE(reference_dev, source_points);
  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, out_point);
  FTB_CHECK_DEVICE(reference_dev, out_index);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_ALL_TYPES(target_points.scalar_type(), "MatchDensePoints",
                          ([&] {
                            MatchPointsDenseKernel<kCUDA, scalar_t> kernel(
                                PointGrid<kCUDA>(target_points, target_mask),
                                KCamera<kCUDA, scalar_t>(kcam),
                                RigidTransform<kCUDA, scalar_t>(rt_cam),
                                source_points, out_point, out_index);

                            Launch1DKernelCUDA(kernel, source_points.size(0));
                          }));
  } else {
    AT_DISPATCH_ALL_TYPES(target_points.scalar_type(), "MatchDensePoints",
                          ([&] {
                            MatchPointsDenseKernel<kCPU, scalar_t> kernel(
                                PointGrid<kCPU>(target_points, target_mask),
                                KCamera<kCPU, scalar_t>(kcam),
                                RigidTransform<kCPU, scalar_t>(rt_cam),
                                source_points, out_point, out_index);

                            Launch1DKernelCPU(kernel, source_points.size(0));
                          }));
  }
}

namespace {
template <Device dev, typename scalar_t>
struct BackwardKernel {
  const RobustCorrespondence<dev, scalar_t> corresp;
  const FeatureMap<dev, scalar_t> tgt_featmap;
  const typename Accessor<dev, scalar_t, 2>::T source_points;
  const typename Accessor<dev, scalar_t, 2>::T source_normals;

  typename Accessor<dev, scalar_t, 2>::T out_points;
  typename Accessor<dev, scalar_t, 2>::T out_normals;
  typename Accessor<dev, scalar_t, 2>::T out_features;
  typename Accessor<dev, bool, 1>::T match_mask;

  BackwardKernel(const torch::Tensor &target_points,
                const torch::Tensor &target_normals,
                const torch::Tensor &target_mask, const torch::Tensor kcam,
                const torch::Tensor &target_features,
                const torch::Tensor &source_points, torch::Tensor out_points,
                torch::Tensor out_normals, torch::Tensor out_features,
                torch::Tensor match_mask)
      : corresp(target_points, target_normals, target_mask, kcam),
        tgt_featmap(target_features),
        source_points(Accessor<dev, scalar_t, 2>::T(source_points)),
        source_normals(Accessor<dev, scalar_t, 2>::T(source_normals)),
        out_points(Accessor<dev, scalar_t, 2>::T(out_points)),
        out_normals(Accessor<dev, scalar_t, 2>::T(out_normals)),
        out_features(Accessor<dev, scalar_t, 2>::T(out_features)),
        match_mask(Accessor<dev, scalar_t, 2>::T(match_mask)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    out_index[idx] = -1;

    const int height = target.points.size(0);
    const int width = target.points.size(1);

    const Vector<scalar_t, 3> point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[idx]));
    
    
    scalar_t du, dv;
    dx_interp.Get(0, du, dv);

    scalar_t j00, j02, j11, j12;
    kcam.Dx_Project(point, j00, j02, j11, j12);
    kcam.Projecti(Tsrc_point, ui, vi);

    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;
    if (target.empty(vi, ui)) return;
    auto target_point = target.points[vi][ui];
    // if (abs(Tsrc_point[2] - target_point[2]) >= 0.07) return;

    out_index[idx] = vi * width + ui;
    out_points[idx][0] = target_point[0];
    out_points[idx][1] = target_point[1];
    out_points[idx][2] = target_point[2];
  }
};

}  // namespace

}  // namespace fiontb
