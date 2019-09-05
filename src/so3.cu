#pragma diagnostic push hd_warning_disable
#pragma hd_warning_disable

#include "so3.hpp"

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


  ExpForwardKernel(const torch::Tensor &x_upsilon_omega, torch::Tensor y_matrix)
      : x_upsilon_omegas(Accessor<dev, scalar_t, 2>::Get(x_upsilon_omega)),
        y_matrix(Accessor<dev, scalar_t, 3>::Get(y_matrix)) {}

  FTB_DEVICE_HOST void operator()(long idx) {
	// Note: this does not implement the real SE3. The translation is
	// not multiplied by the rotation as expected.
	typedef Sophus::SO3<scalar_t> SO3;
    typename SO3::Tangent x_omega;
    x_omega << x_upsilon_omegas[idx][3],
	  x_upsilon_omegas[idx][4], x_upsilon_omegas[idx][5];
    SO3 so3 = SO3::exp(x_omega);
    Eigen::Matrix<scalar_t, 3, 3> matrix = so3.matrix();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        y_matrix[idx][i][j] = matrix(i, j);
      }
    }

    y_matrix[idx][0][3] = x_upsilon_omegas[idx][0];
    y_matrix[idx][1][3] = x_upsilon_omegas[idx][1];
    y_matrix[idx][2][3] = x_upsilon_omegas[idx][2];
  }
};

}  // namespace

torch::Tensor SO3tExpOp::Forward(torch::Tensor x_upsilon_omegas) {
  torch::Tensor matrix =
      torch::empty({x_upsilon_omegas.size(0), 3, 4},
                   torch::TensorOptions(x_upsilon_omegas.type())
                       .device(x_upsilon_omegas.device()));

  AT_DISPATCH_FLOATING_TYPES(
      x_upsilon_omegas.scalar_type(), "ExpForward", ([&] {
        if (x_upsilon_omegas.is_cuda()) {
          ExpForwardKernel<kCUDA, scalar_t> kernel(x_upsilon_omegas, matrix);
          Launch1DKernelCUDA(kernel, x_upsilon_omegas.size(0));
        } else {
          ExpForwardKernel<kCPU, scalar_t> kernel(x_upsilon_omegas, matrix);
          Launch1DKernelCPU(kernel, x_upsilon_omegas.size(0));
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

  FTB_DEVICE_HOST void operator()(long idx) {
    const Matrix3 R(to_matrix<scalar_t, 3, 3>(y_matrices[idx]));
    const Vector3 v(x_upsilon_omegas[idx][3], x_upsilon_omegas[idx][4],
                    x_upsilon_omegas[idx][5]);
    const Matrix3 dy_matrix(to_matrix<scalar_t, 3, 3>(dy_matrices[idx]));
    const scalar_t v_norm = v.squaredNorm();

    const Matrix3 Id_R = (Matrix3::Identity() - R);
    const Matrix3 vv = v * v.transpose();

    for (int k = 0; k < 3; ++k) {
      const Vector3 e(Id_R(0, k), Id_R(1, k), Id_R(2, k));
      Matrix3 skew;

      if (v_norm > 1E-12) {
        const Vector3 v_cross_e = v.cross(e);  // vI row

        const Vector3 vv_ve =
            Vector3(vv(0, k) + v_cross_e[0], vv(1, k) + v_cross_e[1],
                    vv(2, k) + v_cross_e[2]) /
            v_norm;
        skew = SkewMatrix(vv_ve);
      } else {
        switch (k) {
          case 0:
            skew = SkewMatrix(Vector3(1, 0, 0));
            break;
          case 1:
            skew = SkewMatrix(Vector3(0, 1, 0));
            break;
          default:
          case 2:
            skew = SkewMatrix(Vector3(0, 0, 1));
        }
      }

      const Matrix3 rdvk = skew * R;

      const Matrix3 grad = rdvk.array() * dy_matrix.array();
      dx_upsilon_omegas[idx][3 + k] = grad.sum();
    }

	dx_upsilon_omegas[idx][0] = dy_matrices[idx][0][3];
	dx_upsilon_omegas[idx][1] = dy_matrices[idx][1][3];
	dx_upsilon_omegas[idx][2] = dy_matrices[idx][2][3];
  }
};

}  // namespace

torch::Tensor SO3tExpOp::Backward(const torch::Tensor &dy_matrices,
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
          Launch1DKernelCUDA(kernel, size);
        } else {
          ExpBackwardKernel<kCPU, scalar_t> kernel(
              dy_matrices, x_upsilon_omegas, y_matrices, dx_upsilon_omegas);
          Launch1DKernelCPU(kernel, size);
        }
      }));
  return dx_upsilon_omegas;
}

}  // namespace fiontb
