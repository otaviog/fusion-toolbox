#include "icpodometry.hpp"

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {

namespace {

template <typename Kernel>
__global__ void LaunchKernel(Kernel kern) {
  kern();
}

struct CUDAFramebuffer {
  CUDAFramebuffer(const torch::Tensor &points, const torch::Tensor &normals,
                  const torch::Tensor &mask)
      : points(
            points
                .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>()),
        normals(
            normals
                .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>()),
        mask(mask.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits,
                                  size_t>()) {}

  __device__ bool empty(int row, int col) const { return mask[row][col] == 0; }

  const PackedAccessor<float, 3> points;
  const PackedAccessor<float, 3> normals;
  const PackedAccessor<uint8_t, 2> mask;
};

struct GeometricJacobianKernel {
  const CUDAFramebuffer tgt;
  const PackedAccessor<float, 2> src_points;
  const PackedAccessor<uint8_t, 1> src_mask;
  KCamera<kCUDA, float> kcam;
  RTCamera<kCUDA, float> prev_rt_cam;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;

  GeometricJacobianKernel(CUDAFramebuffer tgt,
                          const PackedAccessor<float, 2> src_points,
                          const PackedAccessor<uint8_t, 1> src_mask,
                          KCamera<kCUDA, float> kcam, RTCamera<kCUDA, float> prev_rt_cam,
                          PackedAccessor<float, 2> jacobian,
                          PackedAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_mask(src_mask),
        kcam(kcam),
        prev_rt_cam(prev_rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void operator()() {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
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

    const Eigen::Vector3f p1_on_prev =
        prev_rt_cam.Transform(to_vec3<float>(src_points[ri]));
    Eigen::Vector2i p1_proj = kcam.Project(p1_on_prev);
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

    const Eigen::Vector3f rot_twist = p1_on_prev.cross(normal0);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = (point0 - p1_on_prev).dot(normal0);
  }
};

}  // namespace

void EstimateJacobian_gpu(const torch::Tensor tgt_points,
                          const torch::Tensor tgt_normals,
                          const torch::Tensor tgt_mask,
                          const torch::Tensor src_points,
                          const torch::Tensor src_mask,
                          const torch::Tensor kcam, const torch::Tensor rt_cam,
                          torch::Tensor jacobian, torch::Tensor residual) {
  FTB_CHECK(tgt_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_normals.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_mask.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(src_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(src_mask.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

  GeometricJacobianKernel estm_kern(
      CUDAFramebuffer(tgt_points, tgt_normals, tgt_mask),
      src_points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      src_mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      KCamera<kCUDA, float>(kcam), RTCamera<kCUDA, float>(rt_cam),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>());

  CudaKernelDims kl = Get1DKernelDims(src_points.size(0));
  LaunchKernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

namespace {

__global__ void SobelImageGradient(const PackedAccessor<float, 2> image,
                                   PackedAccessor<float, 3> grad) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = image.size(1);
  const int height = image.size(0);

  if (row >= height || col >= width) return;

  const float Gx[3][3] = {{-1, 0, 1},
						  {-2, 0, 2},
						  {-1, 0, 1}};
  const float Gy[3][3] = {{1, 2, 1},
						  {0, 0, 0},
						  {-1, -2, -1}};

  const int row_start = max(row - 1, 0);
  const int row_end = min(row + 1, height - 1);

  const int col_start = max(col - 1, 0);
  const int col_end = min(col + 1, width - 1);

  float gx_sum = 0.0f;
  float gy_sum = 0.0f;

  for (int irow = row_start; irow <= row_end; ++irow) {
    const int krow = irow - row + 1;
    const auto image_row = image[irow];
    for (int icol = col_start; icol <= col_end; ++icol) {
      const int kcol = icol - col + 1;
	  const float value = image_row[icol];
	  
      gx_sum += value * Gx[krow][kcol];
      gy_sum += value * Gy[krow][kcol];
    }
  }

  grad[row][col][0] = gx_sum;
  grad[row][col][1] = gy_sum;
}

struct CUDAIntensityFramebuffer : public CUDAFramebuffer {
  CUDAIntensityFramebuffer(const torch::Tensor &points,
                           const torch::Tensor &normals,
                           const torch::Tensor &mask,
                           const torch::Tensor &image,
                           const torch::Tensor &image_grad)
      : CUDAFramebuffer(points, normals, mask),
        image(
            image
                .packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()),
        image_grad(
            image_grad.packed_accessor<float, 3, torch::RestrictPtrTraits,
                                       size_t>()) {}

  const PackedAccessor<float, 2> image;
  const PackedAccessor<float, 3> image_grad;
};

struct IntensityJacobianKernel {
  CUDAIntensityFramebuffer tgt;
  const PackedAccessor<float, 2> src_points;
  const PackedAccessor<float, 1> src_intensity;
  const PackedAccessor<uint8_t, 1> src_mask;
  KCamera<kCUDA, float> kcam;
  RTCamera<kCUDA, float> rt_cam;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;

  IntensityJacobianKernel(const CUDAIntensityFramebuffer tgt,
                          const PackedAccessor<float, 2> src_points,
                          const PackedAccessor<float, 1> src_intensity,
                          const PackedAccessor<uint8_t, 1> src_mask,
                          KCamera<kCUDA, float> kcam, RTCamera<kCUDA, float> rt_cam,
                          PackedAccessor<float, 2> jacobian,
                          PackedAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_intensity(src_intensity),
        src_mask(src_mask),
        kcam(kcam),
        rt_cam(rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void operator()() {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
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
        rt_cam.Transform(to_vec3<float>(src_points[ri]));

    int proj_x, proj_y;
    kcam.Projecti(Tsrc_point, proj_x, proj_y);

    if (proj_x < 0 || proj_x >= width || proj_y < 0 || proj_y >= height) return;
    if (tgt.empty(proj_y, proj_x)) return;

    const Eigen::Vector3f tgt_point(to_vec3<float>(tgt.points[proj_y][proj_x]));
    const Eigen::Vector3f tgt_normal(
        to_vec3<float>(tgt.normals[proj_y][proj_x]));

    const float grad_x = tgt.image_grad[proj_y][proj_x][0];
    const float grad_y = tgt.image_grad[proj_y][proj_x][1];

    const Eigen::Matrix<float, 4, 1> dx_kcam(kcam.Dx_Projection(Tsrc_point));
    const Eigen::Vector3f gradk(grad_x * dx_kcam[0], grad_y * dx_kcam[2],
                                grad_x * dx_kcam[1] + grad_y * dx_kcam[3]);

    jacobian[ri][0] = gradk[0];
    jacobian[ri][1] = gradk[1];
    jacobian[ri][2] = gradk[2];

    const Eigen::Vector3f rot_twist = Tsrc_point.cross(gradk);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = tgt.image[proj_y][proj_x] - src_intensity[ri];
  }
};

}  // namespace

void CalcSobelGradient_gpu(const torch::Tensor image, torch::Tensor out_grad) {
  auto grad_acc =
      out_grad.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>();

  const CudaKernelDims kl = Get2DKernelDims(image.size(1), image.size(0));
  SobelImageGradient<<<kl.grid, kl.block>>>(
      image.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      grad_acc);
}

void EstimateIntensityJacobian_gpu(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_intensity, const torch::Tensor src_mask,
    const torch::Tensor kcam, const torch::Tensor rt_cam,
    torch::Tensor jacobian, torch::Tensor residual) {
  FTB_CHECK(tgt_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_normals.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_image.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_grad_image.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_mask.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(src_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(src_intensity.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(src_mask.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

  CUDAIntensityFramebuffer tgt(tgt_points, tgt_normals, tgt_mask, tgt_image,
                               tgt_grad_image);
  IntensityJacobianKernel estm_kern(
      tgt,
      src_points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      src_intensity
          .packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
      src_mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      KCamera<kCUDA, float>(kcam), RTCamera<kCUDA, float>(rt_cam),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>());

  CudaKernelDims kl = Get1DKernelDims(src_points.size(0));
  LaunchKernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb
