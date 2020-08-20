#pragma once

#include "accessor.hpp"
#include "math.hpp"

namespace slamtb {

template <Device dev, typename scalar_t>
struct ExpRt {
  typedef Eigen::Matrix<scalar_t, 3, 3> Matrix33;
  typedef Eigen::Matrix<scalar_t, 3, 4> Matrix34;
  typedef Eigen::Matrix<scalar_t, 3, 1> Vector3;

  Vector3 translation;
  Eigen::AngleAxis<scalar_t> exp_rotation;

  ExpRt(const torch::Tensor &exp_rt) {
    const auto acc = Accessor<kCPU, scalar_t, 1>::Get(exp_rt);

    translation = Vector3(acc[0], acc[1], acc[2]);

    Vector3 v(acc[3], acc[4], acc[5]);
    exp_rotation = Eigen::AngleAxis<scalar_t>(v.norm(), v.normalized());
  }
#pragma nv_exec_check_disable
#pragma hd_warning_disable  
  STB_DEVICE_HOST ExpRt(const typename Accessor<dev, scalar_t, 1>::Ts acc) {
    translation = Vector3(acc[0], acc[1], acc[2]);

    Vector3 v(acc[3], acc[4], acc[5]);
    exp_rotation = Eigen::AngleAxis<scalar_t>(v.norm(), v.normalized());
  }

#pragma nv_exec_check_disable
#pragma hd_warning_disable  
  STB_DEVICE_HOST ExpRt(const typename Accessor<dev, scalar_t, 2>::Ts matrix) {
    translation = Vector3(matrix[0][3], matrix[1][3], matrix[2][3]);
    Matrix33 R(to_matrix<scalar_t, 3, 3>(matrix));
    exp_rotation = Eigen::AngleAxis<scalar_t>(R);
  }

#pragma nv_exec_check_disable
#pragma hd_warning_disable  
  inline STB_DEVICE_HOST Vector3 Transform(const Vector3 &point) const {
    return exp_rotation * point + translation;
  }

#pragma nv_exec_check_disable
#pragma hd_warning_disable  
  inline STB_DEVICE_HOST Matrix34 ToMatrix() const {
    Matrix34 matrix(Matrix34::Zero());
    Matrix33 rot_matrix = exp_rotation.toRotationMatrix();

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix(i, j) = rot_matrix(i, j);
      }
    }

    matrix(0, 3) = translation[0];
    matrix(1, 3) = translation[1];
    matrix(2, 3) = translation[2];

    return matrix;
  }

#pragma nv_exec_check_disable
#pragma hd_warning_disable
  /**
   * This code was based on the
   *
   * https://github.com/abyravan/se3posenets-pytorch/blob/master/layers/SE3ToRt.py
   * from Byravan, Arunkumar, Felix Leeb, Franziska Meier, and
   * Dieter Fox. "Se3-pose-nets: Structured deep dynamics models for
   * visuomotor planning and control." arXiv preprint
   * arXiv:1710.00489 (2017).
   *
   * The derivation of the formula was done by Gallego, Guillermo,
   * and Anthony Yezzi. "A compact formula for the derivative of a
   * 3-D rotation in exponential coordinates." Journal of
   * Mathematical Imaging and Vision 51, no. 3 (2015): 378-384.
   *
   * @param[in] R Rotation matrix part from ToMatrix()
   * @param[out] J The jacobian of the transformation matrix
   * w.r.t. the translation and exponential rotation.
   */
  inline STB_DEVICE_HOST void Dx_ToMatrix(
      const Matrix33 &R, Eigen::Matrix<scalar_t, 12, 6> &J) const {
    J = Eigen::Matrix<scalar_t, 12, 6>::Zero();

    const Vector3 v = exp_rotation.axis() * exp_rotation.angle();
    const scalar_t v_norm = v.squaredNorm();

    const Matrix33 Id_R = (Matrix33::Identity() - R);
    const Matrix33 vv = v * v.transpose();

    for (int k = 0; k < 3; ++k) {
      const Vector3 e(Id_R(0, k), Id_R(1, k), Id_R(2, k));
      Matrix33 skew;

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
      }  // if (v_norm > ...

      const Matrix33 dR_vk = skew * R;

      J(0, 3 + k) = dR_vk(0, 0);
      J(1, 3 + k) = dR_vk(0, 1);
      J(2, 3 + k) = dR_vk(0, 2);

      J(4, 3 + k) = dR_vk(1, 0);
      J(5, 3 + k) = dR_vk(1, 1);
      J(6, 3 + k) = dR_vk(1, 2);

      J(8, 3 + k) = dR_vk(2, 0);
      J(9, 3 + k) = dR_vk(2, 1);
      J(10, 3 + k) = dR_vk(2, 2);
    }

    J(3, 0) = 1;
    J(7, 1) = 1;
    J(11, 2) = 1;
  }
};
}  // namespace slamtb
