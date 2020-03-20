#pragma diagnostic push hd_warning_disable
#pragma hd_warning_disable

#include "se3.hpp"

#include "accessor.hpp"
#include "exp_rt.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
struct ForwardKernel {
  const ExpRt<dev, scalar_t> exp_rt;
  const typename Accessor<dev, scalar_t, 2>::T x_points;
  typename Accessor<dev, scalar_t, 2>::T y_points;

  ForwardKernel(const torch::Tensor &x_exp_rot_t, const torch::Tensor &x_points,
                torch::Tensor y_points)
      : exp_rt(x_exp_rot_t),
        x_points(Accessor<dev, scalar_t, 2>::Get(x_points)),
        y_points(Accessor<dev, scalar_t, 2>::Get(y_points)) {}

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(long idx) {
    const Vector<scalar_t, 3> x_point(to_vec3<scalar_t>(x_points[idx]));
    const auto y_point = exp_rt.Transform(x_point);

    y_points[idx][0] = y_point[0];
    y_points[idx][1] = y_point[1];
    y_points[idx][2] = y_point[2];
  }
};

}  // namespace

void ExpRtTransformOp::Forward(const torch::Tensor &exp_rot_t,
                               const torch::Tensor &x_points,
                               torch::Tensor y_points) {
  AT_DISPATCH_FLOATING_TYPES(x_points.scalar_type(), "ExpForward", ([&] {
                               if (x_points.is_cuda()) {
                                 // ForwardKernel<kCUDA, scalar_t>
                                 // kernel(exp_rot_t, x_points, y_points);
                                 // Launch1DKernelCUDA(kernel,
                                 // x_points.size(0));
                               } else {
                                 ForwardKernel<kCPU, scalar_t> kernel(
                                     exp_rot_t, x_points, y_points);
                                 Launch1DKernelCPU(kernel, x_points.size(0));
                               }
                             }));
}

#pragma nv_exec_check_disable
template <Device dev, typename scalar_t>
FTB_DEVICE_HOST Eigen::Matrix<scalar_t, 6, 1>
ExpRt<dev, scalar_t>::Dx_Transform(
    const ExpRt<dev, scalar_t>::Vector3 &point) const {
  const Eigen::Matrix<scalar_t, 12, 6> J = Dx_ToMatrix();
  const Matrix34 R = ToMatrix();

  Eigen::Matrix<scalar_t, 6, 1> J2;

  // clang-format off
  J2 <<
      J(3, 0), 
      J(7, 1),
      J(11, 2),
      (J(0, 3) * R(0, 0) + J(1, 3) * R(0, 1) + J(2, 3) * R(0, 2) +
       J(4, 3) * R(1, 0) + J(5, 3) * R(1, 1) + J(6, 3) * R(1, 2) +
       J(8, 3) * R(2, 0) + J(9, 3) * R(2, 1) + J(10, 3) * R(2, 2)),
      (J(0, 4) * R(0, 0) + J(1, 4) * R(0, 1) + J(2, 4) * R(0, 2) +
       J(4, 4) * R(1, 0) + J(5, 4) * R(1, 1) + J(6, 4) * R(1, 2) +
       J(8, 4) * R(2, 0) + J(9, 4) * R(2, 1) + J(10, 4) * R(2, 2)),
      (J(0, 5) * R(0, 0) + J(1, 5) * R(0, 1) + J(2, 5) * R(0, 2) +
       J(4, 5) * R(1, 0) + J(5, 5) * R(1, 1) + J(6, 5) * R(1, 2) +
       J(8, 5) * R(2, 0) + J(9, 5) * R(2, 1) + J(10, 5) * R(2, 2));

  // clang-format on
  return J2;
}

namespace {
template <Device dev, typename scalar_t>
struct BackwardKernel {
  const ExpRt<dev, scalar_t> exp_rt;
  const typename Accessor<dev, scalar_t, 2>::T x_points;
  const typename Accessor<dev, scalar_t, 2>::T d_y_points_loss;
  typename Accessor<dev, scalar_t, 2>::T d_exp_rt_loss;

  BackwardKernel(const torch::Tensor &x_exp_rot_t,
                 const torch::Tensor &x_points,
                 const torch::Tensor &d_y_points_loss,
                 torch::Tensor d_exp_rt_loss)
      : exp_rt(x_exp_rot_t),
        x_points(Accessor<dev, scalar_t, 2>::Get(x_points)),
        d_y_points_loss(Accessor<dev, scalar_t, 2>::Get(d_y_points_loss)),
        d_exp_rt_loss(Accessor<dev, scalar_t, 2>::Get(d_exp_rt_loss)) {}

  typedef Eigen::Matrix<scalar_t, 3, 3> Matrix3;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vector3;

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void operator()(long idx) {
    const Vector3 x_point(to_vec3<scalar_t>(x_points[idx]));
    Vector<scalar_t, 3> d_y_loss(to_vec3<scalar_t>(d_y_points_loss[idx]));
#if 0
    const Eigen::Matrix<scalar_t, 6, 1> J = exp_rt.Dx_Transform(x_point);

    Vector<scalar_t, 3> d_y_loss2 = exp_rt.Transform(d_y_loss);

    d_exp_rt_loss[idx][0] = J[0] * d_y_loss[0];
    d_exp_rt_loss[idx][1] = J[1] * d_y_loss[1];
    d_exp_rt_loss[idx][2] = J[2] * d_y_loss[2];
    d_exp_rt_loss[idx][3] = J[3] * d_y_loss2[0];
    d_exp_rt_loss[idx][4] = J[4] * d_y_loss2[1];
    d_exp_rt_loss[idx][5] = J[5] * d_y_loss2[2];
#else
    const Eigen::Matrix<scalar_t, 12, 6> J = exp_rt.Dx_ToMatrix();
    const Eigen::Matrix<scalar_t, 3, 3> mtx = exp_rt.ToMatrix().topLeftCorner(3, 3);
    //d_y_loss = mtx * d_y_loss;

    // 0 1 2
    // 4 5 6
    // 8 9 10

    Eigen::Matrix<scalar_t, 3, 3> dR = SkewMatrix(d_y_loss);
    
    d_exp_rt_loss[idx][0] = J(3, 0) * d_y_loss[0];
    d_exp_rt_loss[idx][1] = J(7, 1) * d_y_loss[1];
    d_exp_rt_loss[idx][2] = J(11, 2) * d_y_loss[2];

#if 1
    d_exp_rt_loss[idx][3] =
        J(0, 3) * dR(0, 2) + J(1, 3) * dR(2, 1) + J(2, 3) * dR(0, 1) +
        J(4, 3) * dR(0, 2) + J(5, 3) * dR(2, 1) + J(6, 3) * dR(0, 1) +
        J(8, 3) * dR(0, 2) + J(9, 3) * dR(2, 1) + J(10, 3) * dR(0, 1);

    d_exp_rt_loss[idx][4] =
        J(0, 4) * dR(0, 2) + J(1, 4) * dR(2, 1) + J(2, 4) * dR(0, 1) +
        J(4, 4) * dR(0, 2) + J(5, 4) * dR(1, 2) + J(6, 4) * dR(0, 1) +
        J(8, 4) * dR(0, 2) + J(9, 4) * dR(1, 2) + J(10, 4) * dR(0, 1);

    d_exp_rt_loss[idx][5] =
        J(0, 5) * dR(0, 2) + J(1, 5) * dR(2, 1) + J(2, 5) * dR(0, 1) +
        J(4, 5) * dR(0, 2) + J(5, 5) * dR(2, 1) + J(6, 5) * dR(0, 1) +
        J(8, 5) * dR(0, 2) + J(9, 5) * dR(2, 1) + J(10, 5) * dR(0, 1);
    
    
#else
    d_exp_rt_loss[idx][3] = (
            J(0, 3) * d_y_loss[1] +  J(4, 3) * d_y_loss[1] +  J(8, 3) * d_y_loss[1] // OK

            
         );

    d_exp_rt_loss[idx][3] =
        J(0, 3) * d_y_loss[1] + J(1, 3) * -d_y_loss[2] +
        J(4, 3) * d_y_loss[1] + J(5, 3) * d_y_loss[2] +
        J(8, 3) * d_y_loss[1] + J(9, 3) * d_y_loss[2];
    d_exp_rt_loss[idx][4] = (
        J(2, 4) * d_y_loss[0] - J(6, 4) * d_y_loss[0] + J(10, 4) * d_y_loss[0] + //OK
        J(2, 4) * d_y_loss[2] + J(6, 4) * d_y_loss[2] - J(10, 4) * d_y_loss[2]
                             );
        
    d_exp_rt_loss[idx][5] = (
        -J(2, 3) * d_y_loss[1] - J(6, 3) * d_y_loss[1] - J(10, 3) * d_y_loss[1] // OK
        
                             );
#endif    

#endif
  }
};

}  // namespace

void ExpRtTransformOp::Backward(const torch::Tensor &x_exp_rot_t,
                                const torch::Tensor &x_points,
                                const torch::Tensor &dy_points,
                                torch::Tensor dx_exp_rot_t) {
  const long size = x_points.size(0);
  AT_DISPATCH_FLOATING_TYPES(x_points.scalar_type(), "ExpBackward", ([&] {
                               if (x_points.is_cuda()) {
                                 // BackwardKernel<kCUDA, scalar_t>
                                 // kernel(x_exp_rot_t, x_points, dy_points,
                                 // dx_exp_rot_t); Launch1DKernelCUDA(kernel,
                                 // size);
                               } else {
                                 BackwardKernel<kCPU, scalar_t> kernel(
                                     x_exp_rot_t, x_points, dy_points,
                                     dx_exp_rot_t);
                                 Launch1DKernelCPU(kernel, size);
                               }
                             }));
}

}  // namespace fiontb
