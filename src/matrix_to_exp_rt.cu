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
  const typename Accessor<dev, scalar_t, 3>::T matrix;
  typename Accessor<dev, scalar_t, 2>::T exp_rt;

  ForwardKernel(const torch::Tensor &matrix, torch::Tensor exp_rt)
      : matrix(Accessor<dev, scalar_t, 3>::Get(matrix)),
        exp_rt(Accessor<dev, scalar_t, 2>::Get(exp_rt)) {}

#pragma nv_exec_check_disable
#pragma hd_warning_disable  
  FTB_DEVICE_HOST void operator()(long idx) {
    const ExpRt<dev, scalar_t> local_exp_rt(matrix[idx]);
    exp_rt[idx][0] = local_exp_rt.translation[0];
    exp_rt[idx][1] = local_exp_rt.translation[1];
    exp_rt[idx][2] = local_exp_rt.translation[2];
    const Vector<scalar_t, 3> v(local_exp_rt.exp_rotation.axis() *
                                local_exp_rt.exp_rotation.angle());
    exp_rt[idx][3] = v[0];
    exp_rt[idx][4] = v[1];
    exp_rt[idx][5] = v[2];
  }
};

}  // namespace

void MatrixToExpRtOp::Forward(const torch::Tensor &matrix,
                              torch::Tensor exp_rt) {
  AT_DISPATCH_FLOATING_TYPES(
      matrix.scalar_type(), "MatrixToExpRtOp::Forward", ([&] {
        if (matrix.is_cuda()) {
          ForwardKernel<kCUDA, scalar_t> kernel(matrix, exp_rt);
          Launch1DKernelCUDA(kernel, matrix.size(0));
        } else {
          ForwardKernel<kCPU, scalar_t> kernel(matrix, exp_rt);
          Launch1DKernelCPU(kernel, matrix.size(0));
        }
      }));
}
}  // namespace slamtb
