#include "kernel_dense_matcher.hpp"

#include "camera.hpp"
#include "error.hpp"

namespace fiontb {
namespace {
__global__ void MatchPointsDense_gpu_kernel(
    const PointGrid<kCUDA> target,
    const Accessor<kCUDA, float, 2>::T src_points, const KCamera<kCUDA> kcam,
    const RTCamera<kCUDA> rt_cam, Accessor<kCUDA, int64_t, 1>::T out_index,
    Accessor<kCUDA, float, 2>::T out_point) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= src_points.size(0)) return;

  out_index[idx] = -1;

  const int width = target.points.size(1);
  const int height = target.points.size(0);

  const Eigen::Vector3f Tsrc_point =
      rt_cam.Transform(to_vec3<float>(src_points[idx]));

  int x, y;
  kcam.Project(Tsrc_point, x, y);
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  if (target.empty(y, x)) return;

  out_index[idx] = y*width + x;
  auto target_point = target.points[y][x];
  out_point[idx][0] = target_point[0];
  out_point[idx][1] = target_point[1];
  out_point[idx][2] = target_point[2];
}
}  // namespace

void MatchDensePoints_gpu(const torch::Tensor target_points,
                          const torch::Tensor target_mask,
                          const torch::Tensor source_points,
                          const torch::Tensor kcam, const torch::Tensor rt_cam,
                          torch::Tensor out_point, torch::Tensor out_index) {
  FTB_CHECK(target_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(source_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(out_index.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(out_point.is_cuda(), "Expected a cuda tensor");

  const CudaKernelDims kl = Get1DKernelDims(source_points.size(0));

  MatchPointsDense_gpu_kernel<<<kl.grid, kl.block>>>(
      PointGrid<kCUDA>(target_points, target_mask),
      Accessor<kCUDA, float, 2>::Get(source_points), KCamera<kCUDA>(kcam),
      RTCamera<kCUDA>(rt_cam), Accessor<kCUDA, int64_t, 1>::Get(out_index),
      Accessor<kCUDA, float, 2>::Get(out_point));
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb