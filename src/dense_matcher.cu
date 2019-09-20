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

    int x, y;
    kcam.Projecti(Tsrc_point, x, y);
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    if (target.empty(x, y)) return;

    out_index[idx] = y * width + x;
    auto target_point = target.points[y][x];
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

void MatchDensePoints(const torch::Tensor target_points,
                      const torch::Tensor target_mask,
                      const torch::Tensor source_points,
                      const torch::Tensor kcam, const torch::Tensor rt_cam,
                      torch::Tensor out_point, torch::Tensor out_index) {
  if (target_points.is_cuda()) {
    FTB_CHECK(source_points.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(out_point.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(out_index.is_cuda(), "Expected a cuda tensor");

    AT_DISPATCH_ALL_TYPES(
        target_points.scalar_type(), "MatchDensePoints", ([&] {
          MatchPointsDenseKernel<kCUDA, scalar_t> kernel(
              PointGrid<kCUDA>(target_points, target_mask),
              KCamera<kCUDA, scalar_t>(kcam), RTCamera<kCUDA, scalar_t>(rt_cam),
              source_points, out_point, out_index);

          LaunchCUDA(kernel, source_points.size(0));
        }));
  } else {
    FTB_CHECK(!source_points.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!kcam.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!rt_cam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(!out_point.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(!out_index.is_cuda(), "Expected a cuda tensor");

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