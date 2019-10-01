#include "matching.hpp"

#include "camera.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "pointgrid.hpp"

namespace fiontb {
namespace {

template <Device dev>
class PointGrid : public BasePointGrid<dev> {
 public:
  PointGrid(const torch::Tensor &points, const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, float, 3>::Get(points)) {}

  typename Accessor<dev, float, 3>::T points;
};

template <Device dev, typename scalar_t>
struct MatchPointsDenseKernel {
  const PointGrid<dev> target;
  const KCamera<dev, scalar_t> kcam;
  const RTCamera<dev, scalar_t> rt_cam;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  typename Accessor<dev, scalar_t, 2>::T out_points;
  typename Accessor<dev, int64_t, 1>::T out_index;

  MatchPointsDenseKernel(const PointGrid<dev> target,
                         const KCamera<dev, scalar_t> kcam,
                         const RTCamera<dev, scalar_t> rt_cam,
                         const torch::Tensor src_points,
                         torch::Tensor out_points,
                         const torch::Tensor out_index)
      : target(target),
        kcam(kcam),
        rt_cam(rt_cam),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        out_points(Accessor<dev, scalar_t, 2>::Get(out_points)),
        out_index(Accessor<dev, int64_t, 1>::Get(out_index)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    if (idx >= src_points.size(0)) return;

    out_index[idx] = -1;

    const int width = target.points.size(1);
    const int height = target.points.size(0);

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[idx]));

    int ui, vi;
    kcam.Projecti(Tsrc_point, ui, vi);
    if (ui < 0 || ui >= width || vi < 0 || vi >= height) return;
    if (target.empty(vi, ui)) return;

    out_index[idx] = vi * width + ui;
    auto target_point = target.points[vi][ui];
    out_points[idx][0] = target_point[0];
    out_points[idx][1] = target_point[1];
    out_points[idx][2] = target_point[2];
  }
};

template <typename scalar_t>
__global__ void ExecKernel(MatchPointsDenseKernel<kCUDA, scalar_t> kernel,
                           int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    kernel(idx);
  }
}

template <typename scalar_t>
inline void LaunchCUDA(MatchPointsDenseKernel<kCUDA, scalar_t> kernel,
                       int size) {
  CudaKernelDims kl = Get1DKernelDims(size);
  ExecKernel<<<kl.grid, kl.block>>>(kernel, size);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename scalar_t>
void LaunchCPU(MatchPointsDenseKernel<kCPU, scalar_t> kernel, int size) {
  for (int i = 0; i < size; ++i) {
    kernel(i);
  }
}

}  // namespace

void Matching::MatchDensePoints(const torch::Tensor target_points,
                      const torch::Tensor target_mask,
                      const torch::Tensor source_points,
                      const torch::Tensor kcam, const torch::Tensor rt_cam,
                      torch::Tensor out_point, torch::Tensor out_index) {
  const auto reference_dev = target_points.device();
  FTB_CHECK_DEVICE(reference_dev, target_points);
  FTB_CHECK_DEVICE(reference_dev, source_points);
  FTB_CHECK_DEVICE(reference_dev, kcam);
  FTB_CHECK_DEVICE(reference_dev, rt_cam);
  FTB_CHECK_DEVICE(reference_dev, out_point);
  FTB_CHECK_DEVICE(reference_dev, out_index);

  if (target_points.is_cuda()) {
    AT_DISPATCH_ALL_TYPES(
        target_points.scalar_type(), "MatchDensePoints", ([&] {
          MatchPointsDenseKernel<kCUDA, scalar_t> kernel(
              PointGrid<kCUDA>(target_points, target_mask),
              KCamera<kCUDA, scalar_t>(kcam), RTCamera<kCUDA, scalar_t>(rt_cam),
              source_points, out_point, out_index);

          LaunchCUDA(kernel, source_points.size(0));
        }));
  } else {
    AT_DISPATCH_ALL_TYPES(
        target_points.scalar_type(), "MatchDensePoints", ([&] {
          MatchPointsDenseKernel<kCPU, scalar_t> kernel(
              PointGrid<kCPU>(target_points, target_mask),
              KCamera<kCPU, scalar_t>(kcam), RTCamera<kCPU, scalar_t>(rt_cam),
              source_points, out_point, out_index);

          LaunchCPU(kernel, source_points.size(0));
        }));
  }
}

}  // namespace fiontb