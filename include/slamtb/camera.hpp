#pragma once

#include <torch/torch.h>
#include "eigen_common.hpp"

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace pybind11 {
class module;
}

namespace slamtb {

/**
 * Intrinsic camera parameters accessor.
 */
template <Device dev, typename scalar_t>
struct KCamera {
  const typename Accessor<dev, scalar_t, 2>::T matrix;

  /**
   * Initializer.
   *
   * @param matrix The [3x3] intrinsic camera matrix. Must be in the
   * same device kind as the `dev` template argument.
   */
  KCamera(const torch::Tensor matrix)
      : matrix(Accessor<dev, scalar_t, 2>::Get(matrix)) {}

  /**
   * Forward project a 3D point into a 2D one, and round to integer.
   *
   * @param point The 3D point.
   */
  FTB_DEVICE_HOST Eigen::Vector2i Project(
      const Vector<scalar_t, 3> point) const {
    const scalar_t img_x = matrix[0][0] * point[0] / point[2] + matrix[0][2];
    const scalar_t img_y = matrix[1][1] * point[1] / point[2] + matrix[1][2];

    return Eigen::Vector2i(round(img_x), round(img_y));
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  /**
   * Forward project a 3D point into a 2D one, and round to integer. V2
   */
  FTB_DEVICE_HOST inline void Projecti(const Vector<scalar_t, 3> point, int &x,
                                       int &y) const {
    scalar_t img_x, img_y;

    Project(point, img_x, img_y);

    x = int(round(img_x));
    y = int(round(img_y));
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  /**
   * Forward
   */
  FTB_DEVICE_HOST inline void Project(const Vector<scalar_t, 3> point,
                                      scalar_t &x, scalar_t &y) const {
    const scalar_t img_x = matrix[0][0] * point[0] / point[2] + matrix[0][2];
    const scalar_t img_y = matrix[1][1] * point[1] / point[2] + matrix[1][2];

    x = img_x;
    y = img_y;
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline void Dx_Project(const Vector<scalar_t, 3> point,
                                         scalar_t &j00, scalar_t &j02,
                                         scalar_t &j11, scalar_t &j12) const {
    const scalar_t fx = matrix[0][0];
    const scalar_t fy = matrix[1][1];

    const scalar_t z = point[2];
    const scalar_t z_sqr = z * z;
    j00 = fx / z;
    j02 = -point[0] * fx / z_sqr;

    j11 = fy / z;
    j12 = -point[1] * fy / z_sqr;
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST Vector<scalar_t, 2> get_center() const {
    return Vector<scalar_t, 2>(matrix[0][2], matrix[1][2]);
  }
};

struct ProjectOp {
  static torch::Tensor Forward(const torch::Tensor &points,
                               const torch::Tensor &intrinsic_matrix);

  static torch::Tensor Backward(const torch::Tensor &dy_grad,
                                const torch::Tensor &points,
                                const torch::Tensor &intrinsic_matrix);

  static void RegisterPybind(pybind11::module &m);
};

template <typename scalar_t>
void InverseTranspose(const Eigen::Matrix<scalar_t, 3, 4, Eigen::DontAlign> &A,
                      Eigen::Matrix<scalar_t, 3, 3, Eigen::DontAlign> &result) {
  scalar_t determinant = +A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
                         A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
                         A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

  scalar_t invdet = 1 / determinant;
  result(0, 0) = (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) * invdet;
  result(1, 0) = -(A(0, 1) * A(2, 2) - A(0, 2) * A(2, 1)) * invdet;
  result(2, 0) = (A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) * invdet;
  result(0, 1) = -(A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) * invdet;
  result(1, 1) = (A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) * invdet;
  result(2, 1) = -(A(0, 0) * A(1, 2) - A(1, 0) * A(0, 2)) * invdet;
  result(0, 2) = (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1)) * invdet;
  result(1, 2) = -(A(0, 0) * A(2, 1) - A(2, 0) * A(0, 1)) * invdet;
  result(2, 2) = (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1)) * invdet;
}

template <typename scalar_t>
struct RigidTransform {
  Eigen::Matrix<scalar_t, 3, 4, Eigen::DontAlign> rt_matrix;
  Eigen::Matrix<scalar_t, 3, 3, Eigen::DontAlign> normal_matrix;

  RigidTransform(const torch::Tensor &matrix) {
    const auto cpu_matrix = matrix.cpu();
    rt_matrix = to_matrix<scalar_t, 3, 4>(cpu_matrix.accessor<scalar_t, 2>());
    const auto rot_cpu_matrix = rt_matrix.topLeftCorner(3, 3);
    InverseTranspose(rt_matrix, normal_matrix);
    // normal_matrix = rot_cpu_matrix.inverse().transpose();
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline Eigen::Matrix<scalar_t, 3, 1> Transform(
      const Eigen::Matrix<scalar_t, 3, 1> &point) const {
    const auto mtx = rt_matrix;
    const scalar_t px = mtx(0, 0) * point[0] + mtx(0, 1) * point[1] +
                        mtx(0, 2) * point[2] + mtx(0, 3);
    const scalar_t py = mtx(1, 0) * point[0] + mtx(1, 1) * point[1] +
                        mtx(1, 2) * point[2] + mtx(1, 3);
    const scalar_t pz = mtx(2, 0) * point[0] + mtx(2, 1) * point[1] +
                        mtx(2, 2) * point[2] + mtx(2, 3);

    return Eigen::Matrix<scalar_t, 3, 1>(px, py, pz);
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline Eigen::Matrix<scalar_t, 3, 1> TransformNormal(
      const Eigen::Matrix<scalar_t, 3, 1> &normal) const {
    const scalar_t nx = normal_matrix(0, 0) * normal[0] +
                        normal_matrix(0, 1) * normal[1] +
                        normal(0, 2) * normal[2];

    const scalar_t ny = normal_matrix(1, 0) * normal[0] +
                        normal_matrix(1, 1) * normal[1] +
                        normal_matrix(1, 2) * normal[2];

    const scalar_t nz = normal_matrix(2, 0) * normal[0] +
                        normal_matrix(2, 1) * normal[1] +
                        normal_matrix(2, 2) * normal[2];

    return Eigen::Matrix<scalar_t, 3, 1>(nx, ny, nz);
  }
};

struct RigidTransformOp {
  static void Rodrigues(const torch::Tensor &rot_matrix,
                        torch::Tensor rodrigues);

  static void TransformPointsInplace(const torch::Tensor &matrix,
                                     torch::Tensor points);

  static void TransformNormalsInplace(const torch::Tensor &matrix,
                                      torch::Tensor normals);

  static void RegisterPybind(pybind11::module &m);
};

}  // namespace slamtb
