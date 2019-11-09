#pragma once

#include <torch/torch.h>
#include "eigen_common.hpp"

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {

template <Device dev, typename scalar_t>
struct KCamera {
  const typename Accessor<dev, scalar_t, 2>::T matrix;

  KCamera(const torch::Tensor matrix)
      : matrix(Accessor<dev, scalar_t, 2>::Get(matrix)) {}

  FTB_DEVICE_HOST Eigen::Vector2i Project(
      const Vector<scalar_t, 3> point) const {
    const scalar_t img_x = matrix[0][0] * point[0] / point[2] + matrix[0][2];
    const scalar_t img_y = matrix[1][1] * point[1] / point[2] + matrix[1][2];

    return Eigen::Vector2i(round(img_x), round(img_y));
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
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

template <Device dev, typename scalar_t>
struct RigidTransform {
  const typename Accessor<dev, scalar_t, 2>::T rt_matrix;

  RigidTransform(const torch::Tensor &rt_matrix)
      : rt_matrix(Accessor<dev, scalar_t, 2>::Get(rt_matrix)) {}

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline Eigen::Matrix<scalar_t, 3, 1> Transform(
      const Eigen::Matrix<scalar_t, 3, 1> &point) const {
    const auto mtx = rt_matrix;
    const scalar_t px = mtx[0][0] * point[0] + mtx[0][1] * point[1] +
                        mtx[0][2] * point[2] + mtx[0][3];
    const scalar_t py = mtx[1][0] * point[0] + mtx[1][1] * point[1] +
                        mtx[1][2] * point[2] + mtx[1][3];
    const scalar_t pz = mtx[2][0] * point[0] + mtx[2][1] * point[1] +
                        mtx[2][2] * point[2] + mtx[2][3];

    return Eigen::Matrix<scalar_t, 3, 1>(px, py, pz);
  }
};

struct RigidTransformOp {
  static void Rodrigues(const torch::Tensor &rot_matrix,
                        torch::Tensor rodrigues);

  static void TransformPoints(const torch::Tensor &matrix,
                              torch::Tensor points);

  static void TransformNormals(const torch::Tensor &matrix,
                               torch::Tensor normals);

  static void RegisterPybind(pybind11::module &m);
};

template <typename scalar_t>
struct RTCamera {
  Eigen::Matrix<scalar_t, 3, 4> rt_matrix;
  Eigen::Matrix<scalar_t, 3, 3> normal_matrix;

  RTCamera(const torch::Tensor &matrix) {
    auto cpu_matrix = matrix.cpu();
    rt_matrix = to_matrix<scalar_t, 3, 4>(cpu_matrix.accessor<scalar_t, 2>());
    normal_matrix = rt_matrix.topLeftCorner(3, 3).inverse().transpose();
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline Eigen::Matrix<scalar_t, 3, 1> Transform(
      const Eigen::Matrix<scalar_t, 3, 1> &point) const {
    const scalar_t px = rt_matrix(0, 0) * point[0] +
                        rt_matrix(0, 1) * point[1] +
                        rt_matrix(0, 2) * point[2] + rt_matrix(0, 3);
    const scalar_t py = rt_matrix(1, 0) * point[0] +
                        rt_matrix(1, 1) * point[1] +
                        rt_matrix(1, 2) * point[2] + rt_matrix(1, 3);
    const scalar_t pz = rt_matrix(2, 0) * point[0] +
                        rt_matrix(2, 1) * point[1] +
                        rt_matrix(2, 2) * point[2] + rt_matrix(2, 3);

    return Eigen::Matrix<scalar_t, 3, 1>(px, py, pz);
  }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  FTB_DEVICE_HOST inline Eigen::Matrix<scalar_t, 3, 1> TransformNormal(
      const Eigen::Matrix<scalar_t, 3, 1> &normal) const {
    return normal_matrix * normal;
  }
};

}  // namespace fiontb
