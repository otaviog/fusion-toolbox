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
    kcamera.Dx_Project(point, j00, j02, j11, j12);

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

void ConcatRTMatrix(const torch::Tensor &left_mtx,
                    const torch::Tensor &right_mtx, torch::Tensor out) {
  AT_DISPATCH_ALL_TYPES(
      left_mtx.scalar_type(), "", ([&] {
        const auto left_acc = left_mtx.accessor<scalar_t, 2>();
        const auto right_acc = right_mtx.accessor<scalar_t, 2>();
        auto out_acc = out.accessor<scalar_t, 2>();
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 4; ++j) {
            out_acc[i][j] = left_acc[i][0] * right_acc[0][j] +
                            left_acc[i][1] * right_acc[1][j] +
                            left_acc[i][2] * right_acc[2][j] + left_acc[i][3];
          }
        }
      }));
}

namespace {
template <Device dev, typename scalar_t>
struct TransformPointsKernel {
  const typename Accessor<dev, scalar_t, 2>::T matrix;
  typename Accessor<dev, scalar_t, 2>::T points;

  TransformPointsKernel(const torch::Tensor &matrix, torch::Tensor points)
      : matrix(Accessor<dev, scalar_t, 2>::Get(matrix)),
        points(Accessor<dev, scalar_t, 2>::Get(points)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const auto point = points[idx];

    const scalar_t x = matrix[0][0] * point[0] + matrix[0][1] * point[1] +
                       matrix[0][2] * point[2] + matrix[0][3];
    const scalar_t y = matrix[1][0] * point[0] + matrix[1][1] * point[1] +
                       matrix[1][2] * point[2] + matrix[1][3];
    const scalar_t z = matrix[2][0] * point[0] + matrix[2][1] * point[1] +
                       matrix[2][2] * point[2] + matrix[2][3];
    points[idx][0] = x;
    points[idx][1] = y;
    points[idx][2] = z;
  }
};
}  // namespace

void RigidTransformOp::TransformPoints(const torch::Tensor &matrix,
                                       torch::Tensor points) {
  const auto ref_device = matrix.device();

  FTB_CHECK_DEVICE(ref_device, points);

  AT_DISPATCH_ALL_TYPES(matrix.scalar_type(), "TransformPoints", [&] {
    if (ref_device.is_cuda()) {
      TransformPointsKernel<kCUDA, scalar_t> kernel(matrix, points);
      Launch1DKernelCUDA(kernel, points.size(0));
    } else {
      TransformPointsKernel<kCPU, scalar_t> kernel(matrix, points);
      Launch1DKernelCPU(kernel, points.size(0));
    }
  });
}

namespace {
template <Device dev, typename scalar_t>
struct TransformNormalsKernel {
  const Eigen::Matrix<scalar_t, 3, 3> matrix;
  typename Accessor<dev, scalar_t, 2>::T normals;

  TransformNormalsKernel(const Eigen::Matrix<scalar_t, 3, 3> matrix,
                         torch::Tensor normals)
      : matrix(matrix), normals(Accessor<dev, scalar_t, 2>::Get(normals)) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const auto normal = normals[idx];

    const scalar_t x = matrix(0, 0) * normal[0] + matrix(0, 1) * normal[1] +
                       matrix(0, 2) * normal[2];
    const scalar_t y = matrix(1, 0) * normal[0] + matrix(1, 1) * normal[1] +
                       matrix(1, 2) * normal[2];
    const scalar_t z = matrix(2, 0) * normal[0] + matrix(2, 1) * normal[1] +
                       matrix(2, 2) * normal[2];
    normals[idx][0] = x;
    normals[idx][1] = y;
    normals[idx][2] = z;
  }
};
}  // namespace

void RigidTransformOp::TransformNormals(const torch::Tensor &matrix,
                                        torch::Tensor normals) {
  const auto ref_device = matrix.device();

  FTB_CHECK_DEVICE(ref_device, normals);

  const auto matrix_cpu = matrix.cpu();
  AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "TransformNormals", [&] {
    const auto cpu_acc = matrix_cpu.accessor<scalar_t, 2>();
    Eigen::Matrix<scalar_t, 3, 3> eigen_matrix;

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) eigen_matrix << cpu_acc[i][j];

    Eigen::Matrix<scalar_t, 3, 3> normal_matrix =
        normal_matrix.inverse().transpose();

    if (ref_device.is_cuda()) {
      TransformNormalsKernel<kCUDA, scalar_t> kernel(normal_matrix, normals);
      Launch1DKernelCUDA(kernel, normals.size(0));
    } else {
      TransformNormalsKernel<kCPU, scalar_t> kernel(normal_matrix, normals);
      Launch1DKernelCPU(kernel, normals.size(0));
    }
  });
}
}  // namespace fiontb
