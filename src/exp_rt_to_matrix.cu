#pragma diagnostic push hd_warning_disable
#pragma hd_warning_disable

#include "se3.hpp"

#include "accessor.hpp"
#include "exp_rt.hpp"
#include "kernel.hpp"

namespace slamtb {
namespace {
template <Device dev, typename scalar_t>
struct ForwardKernel {
  const typename Accessor<dev, scalar_t, 2>::T x_exp_rt;
  typename Accessor<dev, scalar_t, 3>::T y_matrix;

  ForwardKernel(const torch::Tensor &x_exp_rt, torch::Tensor y_matrix)
      : x_exp_rt(Accessor<dev, scalar_t, 2>::Get(x_exp_rt)),
        y_matrix(Accessor<dev, scalar_t, 3>::Get(y_matrix)) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(long idx) {
    const ExpRt<dev, scalar_t> exp_rt(x_exp_rt[idx]);

    const Eigen::Matrix<scalar_t, 3, 4> matrix(exp_rt.ToMatrix());

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        y_matrix[idx][i][j] = matrix(i, j);
      }
    }

    y_matrix[idx][0][3] = x_exp_rt[idx][0];
    y_matrix[idx][1][3] = x_exp_rt[idx][1];
    y_matrix[idx][2][3] = x_exp_rt[idx][2];
  }
};

}  // namespace

void ExpRtToMatrixOp::Forward(const torch::Tensor &exp_rt, torch::Tensor matrix) {
  AT_DISPATCH_FLOATING_TYPES(
      exp_rt.scalar_type(), "ExpRtToMatrixOp::Forward", ([&] {
        if (exp_rt.is_cuda()) {
          ForwardKernel<kCUDA, scalar_t> kernel(exp_rt, matrix);
          Launch1DKernelCUDA(kernel, exp_rt.size(0));
        } else {
          ForwardKernel<kCPU, scalar_t> kernel(exp_rt, matrix);
          Launch1DKernelCPU(kernel, exp_rt.size(0));
        }
      }));
}

namespace {
template <Device dev, typename scalar_t>
struct BackwardKernel {
  const typename Accessor<dev, scalar_t, 3>::T d_R_loss;
  const typename Accessor<dev, scalar_t, 2>::T x_exp_rt;
  const typename Accessor<dev, scalar_t, 3>::T y_matrix;
  typename Accessor<dev, scalar_t, 2>::T d_exp_rt_loss;

  BackwardKernel(const torch::Tensor &d_R_loss, const torch::Tensor &x_exp_rt,
                 const torch::Tensor &y_matrix, torch::Tensor d_exp_rt_loss)
      : d_R_loss(Accessor<dev, scalar_t, 3>::Get(d_R_loss)),
        x_exp_rt(Accessor<dev, scalar_t, 2>::Get(x_exp_rt)),
        y_matrix(Accessor<dev, scalar_t, 3>::Get(y_matrix)),
        d_exp_rt_loss(Accessor<dev, scalar_t, 2>::Get(d_exp_rt_loss)) {}

#pragma nv_exec_check_disable
#pragma hd_warning_disable
  FTB_DEVICE_HOST void operator()(long idx) {
    const ExpRt<dev, scalar_t> exp_rt(x_exp_rt[idx]);
    const auto dR = d_R_loss[idx];

    const Eigen::Matrix<scalar_t, 3, 3> R =
        to_matrix<scalar_t, 3, 3>(y_matrix[idx]);

    Eigen::Matrix<scalar_t, 12, 6> J;
    exp_rt.Dx_ToMatrix(R, J);

    d_exp_rt_loss[idx][0] = J(3, 0) * dR[0][3];
    d_exp_rt_loss[idx][1] = J(7, 1) * dR[1][3];
    d_exp_rt_loss[idx][2] = J(11, 2) * dR[2][3];

    d_exp_rt_loss[idx][3] =
        J(0, 3) * dR[0][0] + J(1, 3) * dR[0][1] + J(2, 3) * dR[0][2] +
        J(4, 3) * dR[1][0] + J(5, 3) * dR[1][1] + J(6, 3) * dR[1][2] +
        J(8, 3) * dR[2][0] + J(9, 3) * dR[2][1] + J(10, 3) * dR[2][2];

    d_exp_rt_loss[idx][4] =
        J(0, 4) * dR[0][0] + J(1, 4) * dR[0][1] + J(2, 4) * dR[0][2] +
        J(4, 4) * dR[1][0] + J(5, 4) * dR[1][1] + J(6, 4) * dR[1][2] +
        J(8, 4) * dR[2][0] + J(9, 4) * dR[2][1] + J(10, 4) * dR[2][2];

    d_exp_rt_loss[idx][5] =
        J(0, 5) * dR[0][0] + J(1, 5) * dR[0][1] + J(2, 5) * dR[0][2] +
        J(4, 5) * dR[1][0] + J(5, 5) * dR[1][1] + J(6, 5) * dR[1][2] +
        J(8, 5) * dR[2][0] + J(9, 5) * dR[2][1] + J(10, 5) * dR[2][2];
  }
};

}  // namespace

void ExpRtToMatrixOp::Backward(const torch::Tensor &d_matrix_loss,
                               const torch::Tensor &x_exp_rt,
                               const torch::Tensor &y_matrix,
                               torch::Tensor d_exp_rt_loss) {
  const long size = x_exp_rt.size(0);
  AT_DISPATCH_FLOATING_TYPES(
      x_exp_rt.scalar_type(), "ExpRtToMatrixOp::Backward", ([&] {
        if (x_exp_rt.is_cuda()) {
          BackwardKernel<kCUDA, scalar_t> kernel(d_matrix_loss, x_exp_rt,
                                                 y_matrix, d_exp_rt_loss);
          Launch1DKernelCUDA(kernel, size);
        } else {
          BackwardKernel<kCPU, scalar_t> kernel(d_matrix_loss, x_exp_rt,
                                                y_matrix, d_exp_rt_loss);
          Launch1DKernelCPU(kernel, size);
        }
      }));
}

}  // namespace slamtb
