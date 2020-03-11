#pragma once

#include "accessor.hpp"
#include "eigen_common.hpp"

namespace fiontb {

namespace {

template <Device dev, typename scalar_t>
struct SE3ICPJacobian {
  typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial;
  typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial;

  FTB_DEVICE_HOST SE3ICPJacobian(
      typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial,
      typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial)
      : JtJ_partial(JtJ_partial), Jtr_partial(Jtr_partial) {
    Zero();
  }

  FTB_DEVICE_HOST void Zero() {
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      Jtr_partial[k] = scalar_t(0);
    }

#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[krow][kcol] = scalar_t(0);
      }
    }
  }

  FTB_DEVICE_HOST void Compute(const Vector<scalar_t, 3> &Tsrc_point,
                               const Vector<scalar_t, 3> &normal,
                               scalar_t residual) {
    scalar_t jacobian[6];
    jacobian[0] = normal[0];
    jacobian[1] = normal[1];
    jacobian[2] = normal[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(normal);
    jacobian[3] = rot_twist[0];
    jacobian[4] = rot_twist[1];
    jacobian[5] = rot_twist[2];

    for (int k = 0; k < 6; ++k) {
      Jtr_partial[k] = jacobian[k] * residual;
    }

#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[krow][kcol] = jacobian[kcol] * jacobian[krow];
      }
    }
  }
};

template <Device dev, typename scalar_t>
struct SO3ICPJacobian {
  typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial;
  typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial;

  FTB_DEVICE_HOST SO3ICPJacobian(
      typename Accessor<dev, scalar_t, 2>::Ts JtJ_partial,
      typename Accessor<dev, scalar_t, 1>::Ts Jtr_partial)
      : JtJ_partial(JtJ_partial), Jtr_partial(Jtr_partial) {
    Zero();
  }

  FTB_DEVICE_HOST void Zero() {
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      Jtr_partial[k] = scalar_t(0);
    }

#pragma unroll
    for (int krow = 0; krow < 6; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 6; ++kcol) {
        JtJ_partial[krow][kcol] = scalar_t(0);
      }
    }
  }

#pragma nv_exec_check_disable
  FTB_DEVICE_HOST void Compute(const Vector<scalar_t, 3> &Tsrc_point,
                               const Vector<scalar_t, 3> &normal,
                               scalar_t residual) {
    scalar_t jacobian[3];
    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(normal);
    jacobian[0] = rot_twist[0];
    jacobian[1] = rot_twist[1];
    jacobian[2] = rot_twist[2];

    for (int k = 0; k < 3; ++k) {
      Jtr_partial[k] = jacobian[k] * residual;
    }

#pragma unroll
    for (int krow = 0; krow < 3; ++krow) {
#pragma unroll
      for (int kcol = 0; kcol < 3; ++kcol) {
        JtJ_partial[krow][kcol] = jacobian[kcol] * jacobian[krow];
      }
    }
  }
};

}  // namespace
}  // namespace fiontb
