#include "kernel_icp_jacobian.hpp"

#include "camera.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "icpodometry.hpp"

namespace fiontb {
namespace {
struct GeometricJacobianKernel {
  CPUPointGrid tgt;
  const torch::TensorAccessor<float, 2> src_points;
  const torch::TensorAccessor<uint8_t, 1> src_mask;
  CPUKCamera kcam;
  CPURTCamera rt_cam;

  torch::TensorAccessor<float, 2> jacobian;
  torch::TensorAccessor<float, 1> residual;

  GeometricJacobianKernel(CPUPointGrid tgt,
                          const torch::TensorAccessor<float, 2> src_points,
                          const torch::TensorAccessor<uint8_t, 1> src_mask,
                          CPUKCamera kcam, CPURTCamera rt_cam,
                          torch::TensorAccessor<float, 2> jacobian,
                          torch::TensorAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_mask(src_mask),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void operator()() {
    for (int ri = 0; ri < src_points.size(0); ri++) {
      Calc(ri);
    }
  }

  void Calc(int ri) {
    if (ri >= src_points.size(0)) return;

    jacobian[ri][0] = 0.0f;
    jacobian[ri][1] = 0.0f;
    jacobian[ri][2] = 0.0f;
    jacobian[ri][3] = 0.0f;
    jacobian[ri][4] = 0.0f;
    jacobian[ri][5] = 0.0f;
    residual[ri] = 0.0f;

    if (src_mask[ri] == 0) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    const Eigen::Vector3f Tsrc_point =
        rt_cam.transform(to_vec3<float>(src_points[ri]));
    Eigen::Vector2i p1_proj = kcam.Project(Tsrc_point);
    if (p1_proj[0] < 0 || p1_proj[0] >= width || p1_proj[1] < 0 ||
        p1_proj[1] >= height)
      return;
    if (tgt.empty(p1_proj[1], p1_proj[0])) return;
    const Eigen::Vector3f point0(
        to_vec3<float>(tgt.points[p1_proj[1]][p1_proj[0]]));
    const Eigen::Vector3f normal0(
        to_vec3<float>(tgt.normals[p1_proj[1]][p1_proj[0]]));

    jacobian[ri][0] = normal0[0];
    jacobian[ri][1] = normal0[1];
    jacobian[ri][2] = normal0[2];

    const Eigen::Vector3f rot_twist = Tsrc_point.cross(normal0);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = (point0 - Tsrc_point).dot(normal0);
  }
};

}  // namespace

void EstimateJacobian_cpu(const torch::Tensor tgt_points,
                          const torch::Tensor tgt_normals,
                          const torch::Tensor tgt_mask,
                          const torch::Tensor src_points,
                          const torch::Tensor src_mask,
                          const torch::Tensor kcam, const torch::Tensor rt_cam,
                          torch::Tensor jacobian, torch::Tensor residual) {
  FTB_CHECK(!tgt_points.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_normals.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_mask.is_cuda(), "Expected a cpu tensor");

  FTB_CHECK(!src_points.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!src_mask.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!kcam.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!rt_cam.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!jacobian.is_cuda(), "Expected a cpu tensor");

  FTB_CHECK(!residual.is_cuda(), "Expected a cpu tensor");

  GeometricJacobianKernel estm_kern(
      CPUPointGrid(tgt_points, tgt_normals, tgt_mask),
      src_points.accessor<float, 2>(), src_mask.accessor<uint8_t, 1>(),
      CPUKCamera(kcam), CPURTCamera(rt_cam), jacobian.accessor<float, 2>(),
      residual.accessor<float, 1>());

  estm_kern();
}

namespace {
struct IntensityJacobianKernel {
  CPUIntensityPointGrid tgt;
  torch::TensorAccessor<float, 2> src_points;
  torch::TensorAccessor<float, 1> src_intensity;
  torch::TensorAccessor<uint8_t, 1> src_mask;

  CPUKCamera kcam;
  CPURTCamera rt_cam;

  torch::TensorAccessor<float, 2> jacobian;
  torch::TensorAccessor<float, 1> residual;

  IntensityJacobianKernel(const CPUIntensityPointGrid tgt,
                          const torch::TensorAccessor<float, 2> src_points,
                          const torch::TensorAccessor<float, 1> src_intensity,
                          const torch::TensorAccessor<uint8_t, 1> src_mask,
                          CPUKCamera kcam, CPURTCamera rt_cam,
                          torch::TensorAccessor<float, 2> jacobian,
                          torch::TensorAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_intensity(src_intensity),
        src_mask(src_mask),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  void operator()() {
#pragma omp parallel for
    for (int ri = 0; ri < src_points.size(0); ri++) {
      Calc(ri);
    }
  }

  void Calc(int ri) {
    jacobian[ri][0] = 0.0f;
    jacobian[ri][1] = 0.0f;
    jacobian[ri][2] = 0.0f;
    jacobian[ri][3] = 0.0f;
    jacobian[ri][4] = 0.0f;
    jacobian[ri][5] = 0.0f;
    residual[ri] = 0.0f;

    if (src_mask[ri] == 0) return;

    const int width = tgt.points.size(1);
    const int height = tgt.points.size(0);

    const Eigen::Vector3f Tsrc_point =
        rt_cam.transform(to_vec3<float>(src_points[ri]));

    int proj_x, proj_y;
    kcam.Project(Tsrc_point, proj_x, proj_y);

    if (proj_x < 0 || proj_x >= width || proj_y < 0 || proj_y >= height) return;
    if (tgt.empty(proj_y, proj_x)) return;

    const Eigen::Vector3f tgt_point(to_vec3<float>(tgt.points[proj_y][proj_x]));
    const Eigen::Vector3f tgt_normal(
        to_vec3<float>(tgt.normals[proj_y][proj_x]));

    const float grad_x = tgt.grad_image[proj_y][proj_x][0];
    const float grad_y = tgt.grad_image[proj_y][proj_x][1];

    const Eigen::Matrix<float, 4, 1> dx_kcam(kcam.Dx_Projection(Tsrc_point));
    const Eigen::Vector3f gradk(grad_x * dx_kcam[0], grad_y * dx_kcam[2],
                                grad_x * dx_kcam[1] + grad_y * dx_kcam[3])*0.05f;

    jacobian[ri][0] = gradk[0];
    jacobian[ri][1] = gradk[1];
    jacobian[ri][2] = gradk[2];

    const Eigen::Vector3f rot_twist = Tsrc_point.cross(gradk);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = src_intensity[ri] - tgt.image[proj_y][proj_x];
  }
};

}  // namespace

void EstimateIntensityJacobian_cpu(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_intensity, const torch::Tensor src_mask,
    const torch::Tensor kcam, const torch::Tensor rt_cam,
    torch::Tensor jacobian, torch::Tensor residual) {
  FTB_CHECK(!tgt_points.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_normals.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_image.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_grad_image.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!tgt_mask.is_cuda(), "Expected a cpu tensor");

  FTB_CHECK(!src_points.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!src_intensity.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!src_mask.is_cuda(), "Expected a cpu tensor");

  FTB_CHECK(!kcam.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!rt_cam.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!jacobian.is_cuda(), "Expected a cpu tensor");
  FTB_CHECK(!residual.is_cuda(), "Expected a cpu tensor");

  CPUIntensityPointGrid tgt(tgt_image, tgt_grad_image, tgt_points, tgt_normals,
                            tgt_mask);
  IntensityJacobianKernel estm_kern(
      tgt, src_points.accessor<float, 2>(), src_intensity.accessor<float, 1>(),
      src_mask.accessor<uint8_t, 1>(), CPUKCamera(kcam), CPURTCamera(rt_cam),
      jacobian.accessor<float, 2>(), residual.accessor<float, 1>());

  estm_kern();
}
}  // namespace fiontb
