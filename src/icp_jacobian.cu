#include "icpodometry.hpp"

#include "accessor.hpp"
#include "camera.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "pointgrid.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
class PointGrid : public BasePointGrid<dev> {
 public:
  const typename Accessor<dev, scalar_t, 3>::T points;
  const typename Accessor<dev, scalar_t, 3>::T normals;

  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
            const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, scalar_t, 3>::Get(points)),
        normals(Accessor<dev, scalar_t, 3>::Get(normals)) {}
};

template <Device dev, typename scalar_t>
struct GeometricJacobianKernel {
  PointGrid<dev, scalar_t> tgt;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, bool, 1>::T src_mask;
  KCamera<dev, scalar_t> kcam;
  RTCamera<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 2>::T jacobian;
  typename Accessor<dev, scalar_t, 1>::T residual;

  GeometricJacobianKernel(PointGrid<dev, scalar_t> tgt,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_mask,
                          KCamera<dev, scalar_t> kcam,
                          RTCamera<dev, scalar_t> rt_cam,
                          torch::Tensor jacobian, torch::Tensor residual)
      : tgt(tgt),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(Accessor<dev, scalar_t, 2>::Get(jacobian)),
        residual(Accessor<dev, scalar_t, 1>::Get(residual)) {}

  FTB_DEVICE_HOST void operator()(int ri) {
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

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));
    Eigen::Vector2i p1_proj = kcam.Project(Tsrc_point);
    if (p1_proj[0] < 0 || p1_proj[0] >= width || p1_proj[1] < 0 ||
        p1_proj[1] >= height)
      return;
    if (tgt.empty(p1_proj[1], p1_proj[0])) return;
    const Vector<scalar_t, 3> point0(
        to_vec3<scalar_t>(tgt.points[p1_proj[1]][p1_proj[0]]));
    const Vector<scalar_t, 3> normal0(
        to_vec3<scalar_t>(tgt.normals[p1_proj[1]][p1_proj[0]]));

    jacobian[ri][0] = normal0[0];
    jacobian[ri][1] = normal0[1];
    jacobian[ri][2] = normal0[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(normal0);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = (point0 - Tsrc_point).dot(normal0);
  }
};

}  // namespace

void ICPJacobian::EstimateGeometric(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_mask, const torch::Tensor kcam,
    const torch::Tensor rt_cam, torch::Tensor jacobian,
    torch::Tensor residual) {
  
  if (src_points.is_cuda()) {
    FTB_CHECK(tgt_points.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_normals.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_mask.is_cuda(), "Expected a cuda tensor");

    FTB_CHECK(src_mask.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");

    FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          GeometricJacobianKernel<kCUDA, scalar_t> kernel(
              PointGrid<kCUDA, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, jacobian, residual);
          Launch1DKernelCUDA(kernel, src_points.size(0));
        });
  } else {
    FTB_CHECK(!tgt_points.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_normals.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_mask.is_cuda(), "Expected a cpu tensor");

    FTB_CHECK(!src_mask.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!kcam.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!rt_cam.is_cuda(), "Expected a cpu tensor");

    FTB_CHECK(!jacobian.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!residual.is_cuda(), "Expected a cpu tensor");

    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "EstimateICPJacobian", [&] {
          GeometricJacobianKernel<kCPU, scalar_t> kernel(
              PointGrid<kCPU, scalar_t>(tgt_points, tgt_normals, tgt_mask),
              src_points, src_mask, kcam, rt_cam, jacobian, residual);
          Launch1DKernelCPU(kernel, src_points.size(0));
        });
  }
}

namespace {
template <Device dev, typename scalar_t>
class IntensityPointGrid : public PointGrid<dev, scalar_t> {
 public:
  IntensityPointGrid(const torch::Tensor &image,
                     const torch::Tensor &grad_image,
                     const torch::Tensor &points, const torch::Tensor normals,
                     const torch::Tensor &mask)
      : PointGrid<dev, scalar_t>(points, normals, mask),
        image(Accessor<dev, scalar_t, 2>::Get(image)),
        grad_image(Accessor<dev, scalar_t, 3>::Get(grad_image)) {}

  typename Accessor<dev, scalar_t, 2>::T image;
  typename Accessor<dev, scalar_t, 3>::T grad_image;
};

template <Device dev, typename scalar_t>
struct IntensityJacobianKernel {
  IntensityPointGrid<dev, scalar_t> tgt;
  const typename Accessor<dev, scalar_t, 2>::T src_points;
  const typename Accessor<dev, scalar_t, 1>::T src_intensity;
  const typename Accessor<dev, bool, 1>::T src_mask;

  KCamera<dev, scalar_t> kcam;
  RTCamera<dev, scalar_t> rt_cam;

  typename Accessor<dev, scalar_t, 2>::T jacobian;
  typename Accessor<dev, scalar_t, 1>::T residual;

  IntensityJacobianKernel(const IntensityPointGrid<dev, scalar_t> tgt,
                          const torch::Tensor &src_points,
                          const torch::Tensor &src_intensity,
                          const torch::Tensor &src_mask,
                          KCamera<dev, scalar_t> kcam,
                          RTCamera<dev, scalar_t> rt_cam,
                          torch::Tensor jacobian, torch::Tensor residual)
      : tgt(tgt),
        src_points(Accessor<dev, scalar_t, 2>::Get(src_points)),
        src_intensity(Accessor<dev, scalar_t, 1>::Get(src_intensity)),
        src_mask(Accessor<dev, bool, 1>::Get(src_mask)),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(Accessor<dev, scalar_t, 2>::Get(jacobian)),
        residual(Accessor<dev, scalar_t, 1>::Get(residual)) {}

  FTB_DEVICE_HOST void operator()(int ri) {
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

    const Vector<scalar_t, 3> Tsrc_point =
        rt_cam.Transform(to_vec3<scalar_t>(src_points[ri]));

    int proj_x, proj_y;
    kcam.Projecti(Tsrc_point, proj_x, proj_y);

    if (proj_x < 0 || proj_x >= width || proj_y < 0 || proj_y >= height) return;
    if (tgt.empty(proj_y, proj_x)) return;

    const Vector<scalar_t, 3> tgt_point(
        to_vec3<scalar_t>(tgt.points[proj_y][proj_x]));
    const Vector<scalar_t, 3> tgt_normal(
        to_vec3<scalar_t>(tgt.normals[proj_y][proj_x]));

    const scalar_t grad_x = tgt.grad_image[proj_y][proj_x][0];
    const scalar_t grad_y = tgt.grad_image[proj_y][proj_x][1];

    const Eigen::Matrix<scalar_t, 4, 1> dx_kcam(kcam.Dx_Projection(Tsrc_point));
    const Vector<scalar_t, 3> gradk(grad_x * dx_kcam[0], grad_y * dx_kcam[2],
                                    grad_x * dx_kcam[1] + grad_y * dx_kcam[3]);

    jacobian[ri][0] = gradk[0];
    jacobian[ri][1] = gradk[1];
    jacobian[ri][2] = gradk[2];

    const Vector<scalar_t, 3> rot_twist = Tsrc_point.cross(gradk);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = src_intensity[ri] - tgt.image[proj_y][proj_x];
  }
};

}  // namespace

void ICPJacobian::EstimateIntensity(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_intensity, const torch::Tensor src_mask,
    const torch::Tensor kcam, const torch::Tensor rt_cam,
    torch::Tensor jacobian, torch::Tensor residual) {
  if (src_points.is_cuda()) {
    FTB_CHECK(tgt_points.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_normals.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_image.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_grad_image.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(tgt_mask.is_cuda(), "Expected a cuda tensor");

    FTB_CHECK(src_intensity.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(src_mask.is_cuda(), "Expected a cuda tensor");

    FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "IntensityJacobianKernel", ([&] {
          IntensityPointGrid<kCUDA, scalar_t> tgt(
              tgt_image, tgt_grad_image, tgt_points, tgt_normals, tgt_mask);
          IntensityJacobianKernel<kCUDA, scalar_t> kernel(
              tgt, src_points, src_intensity, src_mask,
              KCamera<kCUDA, scalar_t>(kcam), RTCamera<kCUDA, scalar_t>(rt_cam),
              jacobian, residual);
          Launch1DKernelCUDA(kernel, src_points.size(0));
        }));
  } else {
    FTB_CHECK(!tgt_points.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_normals.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_image.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_grad_image.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!tgt_mask.is_cuda(), "Expected a cpu tensor");

    FTB_CHECK(!src_intensity.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!src_mask.is_cuda(), "Expected a cpu tensor");

    FTB_CHECK(!kcam.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!rt_cam.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!jacobian.is_cuda(), "Expected a cpu tensor");
    FTB_CHECK(!residual.is_cuda(), "Expected a cpu tensor");

    AT_DISPATCH_FLOATING_TYPES(
        src_points.scalar_type(), "IntensityJacobianKernel", [&] {
          IntensityPointGrid<kCPU, scalar_t> tgt(
              tgt_image, tgt_grad_image, tgt_points, tgt_normals, tgt_mask);
          IntensityJacobianKernel<kCPU, scalar_t> kernel(
              tgt, src_points, src_intensity, src_mask,
              KCamera<kCPU, scalar_t>(kcam), RTCamera<kCPU, scalar_t>(rt_cam),
              jacobian, residual);
          Launch1DKernelCPU(kernel, src_points.size(0));
        });
  }
}
}  // namespace fiontb
