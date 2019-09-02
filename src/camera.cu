#include "camera.hpp"

#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {

namespace {

template <Device dev, typename scalar_t>
struct ProjectKernel {
  const typename Accessor<dev, scalar_t, 2>::T points;
  typename Accessor<dev, scalar_t, 2>::T out_projection;
  const KCamera<dev, scalar_t> kcamera;

  ProjectKernel(const torch::Tensor &points, const torch::Tensor &intrinsics,
                torch::Tensor out_projection)
      : points(Accessor<dev, scalar_t, 2>::Get(points)),
        kcamera(intrinsics),
        out_projection(Accessor<dev, scalar_t, 2>::Get(out_projection)) {}

  FTB_DEVICE_HOST void operator()(int i) {
    const Vector<scalar_t, 3> point = to_vec3<scalar_t>(points[i]);
    scalar_t x, y;
    kcamera.Project(point, x, y);

    out_projection[i][0] = x;
    out_projection[i][1] = y;
  }
};

}  // namespace

torch::Tensor ProjectOp::Forward(const torch::Tensor &points,
                                 const torch::Tensor &intrinsics) {
  auto bk_points = points.view({-1, 3});

  const torch::Tensor out_projection =
      torch::empty(bk_points.sizes(),
                   torch::TensorOptions(points.type()).device(points.device()));

  if (points.is_cuda()) {
    FTB_CHECK(intrinsics.is_cuda(), "Expected a CUDA tensor");
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project", ([&] {
                            ProjectKernel<kCUDA, scalar_t> kernel(
                                bk_points, intrinsics, out_projection);
                            Launch1DKernel<kCUDA>(kernel, bk_points.size(0));
                          }));
  } else {
    FTB_CHECK(!intrinsics.is_cuda(), "Expected a CPU tensor");
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project", ([&] {
                            ProjectKernel<kCPU, scalar_t> kernel(
                                bk_points, intrinsics, out_projection);
                            Launch1DKernel<kCPU>(kernel, bk_points.size(0));
                          }));
  }

  return out_projection;
}

namespace {
template <Device dev, typename scalar_t>
struct ProjectBackwardKernel {
  const typename Accessor<dev, scalar_t, 2>::T dy_grad, points;
  const KCamera<dev, scalar_t> kcamera;
  typename Accessor<dev, scalar_t, 2>::T dx_points;

  ProjectBackwardKernel(const torch::Tensor &dy_grad,
                        const torch::Tensor &points,
                        const torch::Tensor &intrinsics,
                        torch::Tensor dx_points)
      : dy_grad(Accessor<dev, scalar_t, 2>::Get(dy_grad)),
        points(Accessor<dev, scalar_t, 2>::Get(points)),
        kcamera(intrinsics),
        dx_points(Accessor<dev, scalar_t, 2>::Get(dx_points)) {}

  FTB_DEVICE_HOST void operator()(int i) {
    Eigen::Matrix<scalar_t, 3, 1> point = to_vec3<scalar_t>(points[i]);

    const scalar_t focal_len_x = kcamera.kcam_matrix[0][0];
    const scalar_t focal_len_y = kcamera.kcam_matrix[1][1];

    const scalar_t zsqr = point[2] * point[2];

    const scalar_t J00 = focal_len_x / point[2];
    const scalar_t J02 = -point[0] * focal_len_x / zsqr;

    const scalar_t J11 = focal_len_y / point[2];
    const scalar_t J12 = -point[1] * focal_len_y / zsqr;

    dx_points[i][0] = J00 * dy_grad[i][0];
    dx_points[i][1] = J11 * dy_grad[i][1];
    dx_points[i][2] = J02 * dy_grad[i][0] + J12 * dy_grad[i][1];
  }
};
}  // namespace

torch::Tensor ProjectOp::Backward(const torch::Tensor &dy_grad,
                                  const torch::Tensor &points,
                                  const torch::Tensor &intrinsics) {
  auto bk_points = points.view({-1, 3});
  const torch::Tensor dx_points =
      torch::empty(bk_points.sizes(),
                   torch::TensorOptions(points.type()).device(points.device()));

  if (points.is_cuda()) {
    FTB_CHECK(intrinsics.is_cuda(), "Expected a CUDA tensor");
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project", ([&] {
                            ProjectBackwardKernel<kCUDA, scalar_t> kernel(
                                dy_grad, bk_points, intrinsics, dx_points);
                            Launch1DKernel<kCUDA>(kernel, bk_points.size(0));
                          }));
  } else {
    FTB_CHECK(!intrinsics.is_cuda(), "Expected a CPU tensor");
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project", ([&] {
                            ProjectBackwardKernel<kCPU, scalar_t> kernel(
                                dy_grad, bk_points, intrinsics, dx_points);
                            Launch1DKernel<kCPU>(kernel, bk_points.size(0));
                          }));
  }

  return dx_points;
}

}  // namespace fiontb
