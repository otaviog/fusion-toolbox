#pragma once

#include <torch/torch.h>
#include "eigen_common.hpp"

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace fiontb {
template <bool CUDA>
struct KCamera {
  KCamera(torch::Tensor kcam_matrix)
      : kcam_matrix(Accessor<CUDA, float, 2>::Get(kcam_matrix)) {}
  FTB_DEVICE_HOST Eigen::Vector2i Project(const Eigen::Vector3f point) {
    const float img_x =
        kcam_matrix[0][0] * point[0] / point[2] + kcam_matrix[0][2];
    const float img_y =
        kcam_matrix[1][1] * point[1] / point[2] + kcam_matrix[1][2];

    return Eigen::Vector2i(round(img_x), round(img_y));
  }

  FTB_DEVICE_HOST void Project(const Eigen::Vector3f point, int &x, int &y) {
    const float img_x =
        kcam_matrix[0][0] * point[0] / point[2] + kcam_matrix[0][2];
    const float img_y =
        kcam_matrix[1][1] * point[1] / point[2] + kcam_matrix[1][2];

    x = round(img_x);
    y = round(img_y);
  }

  FTB_DEVICE_HOST Eigen::Matrix<float, 4, 1> Dx_Projection(
      const Eigen::Vector3f point) {
    Eigen::Matrix<float, 4, 1> coeffs;

    const float fx = kcam_matrix[0][0];
    const float fy = kcam_matrix[1][1];

    const float z = point[2];
    const float z_sqr = z * z;
    coeffs << fx / z, -point[0] * fx / z_sqr, fy / z, -point[1] * fy / z_sqr;
    return coeffs;
  }
  const typename Accessor<CUDA, float, 2>::Type kcam_matrix;
};

typedef KCamera<false> CPUKCamera;
typedef KCamera<true> CUDAKCamera;

template <bool CUDA>
struct RTCamera {
  RTCamera(torch::Tensor rt_matrix)
      : rt_matrix(Accessor<CUDA, float, 2>::Get(rt_matrix)) {}

  FTB_DEVICE_HOST Eigen::Vector3f transform(const Eigen::Vector3f point) const {
    const auto mtx = rt_matrix;
    const float px = mtx[0][0] * point[0] + mtx[0][1] * point[1] +
                     mtx[0][2] * point[2] + mtx[0][3];
    const float py = mtx[1][0] * point[0] + mtx[1][1] * point[1] +
                     mtx[1][2] * point[2] + mtx[1][3];
    const float pz = mtx[2][0] * point[0] + mtx[2][1] * point[1] +
                     mtx[2][2] * point[2] + mtx[2][3];

    return Eigen::Vector3f(px, py, pz);
  }
  const typename Accessor<CUDA, float, 2>::Type rt_matrix;
};

typedef RTCamera<false> CPURTCamera;
typedef RTCamera<true> CUDARTCamera;

}  // namespace fiontb
