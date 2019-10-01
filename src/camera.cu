#include "camera.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

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
      torch::empty({bk_points.size(0), 2},
                   torch::TensorOptions(points.type()).device(points.device()));

  const auto reference_dev = points.device();
  FTB_CHECK_DEVICE(reference_dev, intrinsics);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project.forward", ([&] {
                            ProjectKernel<kCUDA, scalar_t> kernel(
                                bk_points, intrinsics, out_projection);
                            Launch1DKernelCUDA(kernel, bk_points.size(0));
                          }));
  } else {
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project.forward", ([&] {
                            ProjectKernel<kCPU, scalar_t> kernel(
                                bk_points, intrinsics, out_projection);
                            Launch1DKernelCPU(kernel, bk_points.size(0));
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

    scalar_t j00, j02, j11, j12;
    kcamera.Dx_Projection(point, j00, j02, j11, j12);

    dx_points[i][0] = j00 * dy_grad[i][0];
    dx_points[i][1] = j11 * dy_grad[i][1];
    dx_points[i][2] = j02 * dy_grad[i][0] + j12 * dy_grad[i][1];
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

  const auto reference_dev = points.device();
  FTB_CHECK_DEVICE(reference_dev, intrinsics);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project.backward", ([&] {
                            ProjectBackwardKernel<kCUDA, scalar_t> kernel(
                                dy_grad, bk_points, intrinsics, dx_points);
                            Launch1DKernelCUDA(kernel, bk_points.size(0));
                          }));
  } else {
    AT_DISPATCH_ALL_TYPES(points.scalar_type(), "Project.backward", ([&] {
                            ProjectBackwardKernel<kCPU, scalar_t> kernel(
                                dy_grad, bk_points, intrinsics, dx_points);
                            Launch1DKernelCPU(kernel, bk_points.size(0));
                          }));
  }

  return dx_points;
}

void ProjectOp::RegisterPybind(pybind11::module &m) {
  m.def("project_op_forward", &ProjectOp::Forward);
  m.def("project_op_backward", &ProjectOp::Backward);
}

}  // namespace fiontb
