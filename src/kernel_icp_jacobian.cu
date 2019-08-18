#include "icpodometry.hpp"

#include "cuda_camera.hpp"
#include "cuda_utils.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {

namespace {

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
  CUDAKCamera kcam;
  CUDARTCamera prev_rt_cam;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;

  GeometricJacobianKernel(CUDAFramebuffer tgt,
                          const PackedAccessor<float, 2> src_points,
                          const PackedAccessor<uint8_t, 1> src_mask,
                          CUDAKCamera kcam, CUDARTCamera prev_rt_cam,
                          PackedAccessor<float, 2> jacobian,
                          PackedAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_mask(src_mask),
        kcam(kcam),
        prev_rt_cam(prev_rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void EstimateJacobian() {
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
        prev_rt_cam.transform(to_vec3<float>(src_points[ri]));
    Eigen::Vector2i p1_proj = kcam.project(p1_on_prev);
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

__global__ void EstimateJacobian_gpu_kernel(GeometricJacobianKernel kernel) {
  kernel.EstimateJacobian();
}
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
      CUDAKCamera(kcam), CUDARTCamera(rt_cam),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>());

  CudaKernelDims kl = Get1DKernelDims(src_points.size(0));
  EstimateJacobian_gpu_kernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

namespace {

__device__ float SquaredNorm(
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, size_t>
        vec0,
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, size_t>
        vec1) {
  float sum = 0.0f;
  for (int i = 0; i < vec0.size(0); ++i) {
    const float diff = (vec1[i] - vec0[i]);
    sum += diff * diff;
  }
  return sum;
}

__global__ void SobelDescriptorGradient(const PackedAccessor<float, 3> image,
                                        PackedAccessor<float, 3> grad) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = image.size(1);
  const int height = image.size(0);

  if (row >= height || col >= width) return;

  const float Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const float Gy[3][3] = {{-1, -2, 1}, {0, 0, 0}, {1, 2, 1}};

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
      gx_sum += image_row[icol][0] * Gx[krow][kcol];
      gy_sum += image_row[icol][0] * Gy[krow][kcol];
    }
  }

  grad[row][col][0] = gx_sum;
  grad[row][col][1] = gy_sum;
}

struct CUDADescriptorFramebuffer : public CUDAFramebuffer {
  CUDADescriptorFramebuffer(const torch::Tensor &points,
                            const torch::Tensor &normals,
                            const torch::Tensor &mask,
                            const torch::Tensor &descriptors)
      : CUDAFramebuffer(points, normals, mask),
        descriptors(
            descriptors.packed_accessor<float, 3, torch::RestrictPtrTraits,
                                        size_t>()) {}

  const PackedAccessor<float, 3> descriptors;
};

struct DescriptorJacobianKernel {
  CUDADescriptorFramebuffer tgt;
  const PackedAccessor<float, 3> grad_xy;
  const PackedAccessor<float, 2> src_points;
  const PackedAccessor<float, 2> src_descriptors;
  const PackedAccessor<uint8_t, 1> src_mask;
  CUDAKCamera kcam;
  CUDARTCamera prev_rt_cam;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;

  DescriptorJacobianKernel(const CUDADescriptorFramebuffer tgt,
                           const PackedAccessor<float, 2> src_points,
                           const PackedAccessor<float, 2> src_descriptors,
                           const PackedAccessor<uint8_t, 1> src_mask,
                           const PackedAccessor<float, 3> grad_xy,
                           CUDAKCamera kcam, CUDARTCamera prev_rt_cam,
                           PackedAccessor<float, 2> jacobian,
                           PackedAccessor<float, 1> residual)
      : tgt(tgt),
        src_points(src_points),
        src_descriptors(src_descriptors),
        src_mask(src_mask),
        grad_xy(grad_xy),
        kcam(kcam),
        prev_rt_cam(prev_rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void EstimateJacobian() {
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
        prev_rt_cam.transform(to_vec3<float>(src_points[ri]));
    Eigen::Vector2i p1_proj = kcam.project(p1_on_prev);
    int proj_x = p1_proj[0], proj_y = p1_proj[1];

    if (proj_x < 0 || proj_x >= width || proj_y < 0 || proj_y >= height) return;
    if (tgt.empty(proj_y, proj_x)) return;

    const auto tgt_desc = tgt.descriptors[proj_y][proj_x];
    const auto src_desc = src_descriptors[ri];

    const Eigen::Vector3f tgt_point(to_vec3<float>(tgt.points[proj_y][proj_x]));
    const Eigen::Vector3f tgt_normal(
        to_vec3<float>(tgt.normals[proj_y][proj_x]));

    const float desc_grad_x = grad_xy[proj_y][proj_x][0];
    const float desc_grad_y = grad_xy[proj_y][proj_x][1];

    const Eigen::Matrix<float, 4, 1> dx_kcam(kcam.Dx_projection(p1_on_prev));
    const Eigen::Vector3f gradk(
        desc_grad_x * dx_kcam[0], desc_grad_y * dx_kcam[2],
        desc_grad_x * dx_kcam[1] + desc_grad_y * dx_kcam[3]);

    jacobian[ri][0] = gradk[0];
    jacobian[ri][1] = gradk[1];
    jacobian[ri][2] = gradk[2];

    const Eigen::Vector3f rot_twist = p1_on_prev.cross(gradk);
    jacobian[ri][3] = rot_twist[0];
    jacobian[ri][4] = rot_twist[1];
    jacobian[ri][5] = rot_twist[2];

    residual[ri] = SquaredNorm(tgt_desc, src_desc);
  }
};

__global__ void EstimateDescriptorJacobian_gpu_kernel(
    DescriptorJacobianKernel kernel) {
  kernel.EstimateJacobian();
}
}  // namespace

void EstimateDescriptorJacobian_gpu(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_descriptors, const torch::Tensor tgt_mask,
    const torch::Tensor src_points, const torch::Tensor src_descriptors,
    const torch::Tensor src_mask, const torch::Tensor kcam,
    const torch::Tensor rt_cam, torch::Tensor jacobian,
    torch::Tensor residual) {
  FTB_CHECK(tgt_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_normals.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_descriptors.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(tgt_mask.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(src_points.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(src_descriptors.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(src_mask.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

  torch::Tensor descriptor_grad = torch::empty(
      {tgt_points.size(0), tgt_points.size(1)},
      torch::TensorOptions(torch::kFloat).device(tgt_points.device()));
  auto descr_grad_acc =
      descriptor_grad
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>();

  CudaKernelDims kl = Get2DKernelDims(tgt_points.size(1), tgt_points.size(0));
  SobelDescriptorGradient<<<kl.grid, kl.block>>>(
      tgt_descriptors
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      descr_grad_acc);

  CUDADescriptorFramebuffer tgt(tgt_points, tgt_normals, tgt_mask,
                                tgt_descriptors);
  DescriptorJacobianKernel estm_kern(
      tgt,
      src_points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      src_descriptors
          .packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      src_mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      descr_grad_acc, CUDAKCamera(kcam), CUDARTCamera(rt_cam),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>());

  kl = Get1DKernelDims(src_points.size(0));
  EstimateDescriptorJacobian_gpu_kernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb
