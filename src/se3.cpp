#include "se3.hpp"

#include <sophus/se3.hpp>
#include "eigen_common.hpp"

#include "accessor.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {
namespace {
template <Device dev, typename scalar_t>
struct ExpForwardKernel {
  const typename Accessor<dev, scalar_t, 2>::T x_upsilon_omegas;
  typename Accessor<dev, scalar_t, 3>::T y_matrix;

  typedef Sophus::SE3<scalar_t> SE3;

  ExpForwardKernel(const torch::Tensor &x_upsilon_omega, torch::Tensor y_matrix)
      : x_upsilon_omegas(Accessor<dev, scalar_t, 2>::Get(x_upsilon_omega)),
        y_matrix(Accessor<dev, scalar_t, 3>::Get(y_matrix)) {}

  void operator()(long idx) {
    typename SE3::Tangent x_upsilon_omega;
    x_upsilon_omega << x_upsilon_omegas[idx][0], x_upsilon_omegas[idx][1],
        x_upsilon_omegas[idx][2], x_upsilon_omegas[idx][3],
        x_upsilon_omegas[idx][4], x_upsilon_omegas[idx][5];
    SE3 se3 = SE3::exp(x_upsilon_omega);
    Eigen::Matrix<scalar_t, 3, 4> matrix = se3.matrix3x4();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        y_matrix[idx][i][j] = matrix(i, j);
      }
    }
  }
};

}  // namespace

torch::Tensor SE3ExpOp::Forward(torch::Tensor x_upsilon_omegas) {
  torch::Tensor matrix = torch::empty(
      {x_upsilon_omegas.size(0), 3, 4},
      torch::TensorOptions(x_upsilon_omegas.type())
      .device(x_upsilon_omegas.device()));

  AT_DISPATCH_FLOATING_TYPES(
      x_upsilon_omegas.scalar_type(), "ExpForward", ([&] {
        if (x_upsilon_omegas.is_cuda()) {
          ExpForwardKernel<kCUDA, scalar_t> kernel(x_upsilon_omegas, matrix);
          Launch1DKernel<kCUDA>(kernel, x_upsilon_omegas.size(0));
        } else {
          ExpForwardKernel<kCPU, scalar_t> kernel(x_upsilon_omegas, matrix);
          Launch1DKernel<kCPU>(kernel, x_upsilon_omegas.size(0));
        }
      }));
  return matrix;
}

namespace {
template <Device dev, typename scalar_t>
struct ExpBackwardKernel {
  const typename Accessor<dev, scalar_t, 3>::T dy_matrices;
  const typename Accessor<dev, scalar_t, 2>::T x_upsilon_omegas;
  const typename Accessor<dev, scalar_t, 3>::T y_matrices;
  typename Accessor<dev, scalar_t, 2>::T dx_upsilon_omegas;

  ExpBackwardKernel(const torch::Tensor &dy_matrices,
                    const torch::Tensor &x_upsilon_omegas,
                    const torch::Tensor &y_matrices,
                    torch::Tensor dx_upsilon_omegas)
      : dy_matrices(Accessor<dev, scalar_t, 3>::Get(dy_matrices)),
        x_upsilon_omegas(Accessor<dev, scalar_t, 2>::Get(x_upsilon_omegas)),
        y_matrices(Accessor<dev, scalar_t, 3>::Get(y_matrices)),
        dx_upsilon_omegas(Accessor<dev, scalar_t, 2>::Get(dx_upsilon_omegas)) {}

  typedef Eigen::Matrix<scalar_t, 3, 3> Matrix3;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vector3;

  void operator()(long idx) {
    const Matrix3 R(to_matrix<scalar_t, 3, 3>(y_matrices[idx]));
    const Vector3 v(x_upsilon_omegas[idx][3], x_upsilon_omegas[idx][4],
                    x_upsilon_omegas[idx][5]);
    const Matrix3 dy_matrix(to_matrix<scalar_t, 3, 3>(dy_matrices[idx]));
    const scalar_t v_norm = v.norm();

    const Matrix3 Id_R = (Matrix3::Identity() - R);
    const Matrix3 V = v * v.transpose();

    for (int k = 0; k < 3; ++k) {
      const Vector3 e(Id_R(k, 0), Id_R(k, 1), Id_R(k, 2));

      const Matrix3 skew_v(v[k] * SkewMatrix(v));
      const Vector3 vxe = v.cross(e);
      const Matrix3 skew_vxe(SkewMatrix(vxe));

      const Matrix3 rdvk = ((skew_v + skew_vxe) / v_norm) * R;

      const Matrix3 grad = rdvk.array() * dy_matrix.array();
      dx_upsilon_omegas[idx][k] = grad.sum();
    }
  }
};

}  // namespace

torch::Tensor SE3ExpOp::Backward(const torch::Tensor &dy_matrices,
                                 const torch::Tensor &x_upsilon_omegas,
                                 const torch::Tensor &y_matrices) {
  torch::Tensor dx_upsilon_omegas = torch::empty(
      x_upsilon_omegas.sizes(), torch::TensorOptions(x_upsilon_omegas.type())
                                    .device(x_upsilon_omegas.device()));

  const long size = x_upsilon_omegas.size(0);
  AT_DISPATCH_FLOATING_TYPES(
      x_upsilon_omegas.scalar_type(), "ExpBackward", ([&] {
        if (x_upsilon_omegas.is_cuda()) {
          ExpBackwardKernel<kCUDA, scalar_t> kernel(
              dy_matrices, x_upsilon_omegas, y_matrices, dx_upsilon_omegas);
          Launch1DKernel<kCUDA>(kernel, size);
        } else {
          ExpBackwardKernel<kCPU, scalar_t> kernel(
              dy_matrices, x_upsilon_omegas, y_matrices, dx_upsilon_omegas);
          Launch1DKernel<kCPU>(kernel, size);
        }
      }));
  return dx_upsilon_omegas;
}

}  // namespace fiontb
