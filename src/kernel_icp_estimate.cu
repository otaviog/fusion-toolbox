#include "icpodometry.hpp"

#include "cuda_utils.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {
struct KCamera {
  KCamera(torch::Tensor kcam_matrix)
      : kcam_matrix(
            kcam_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits,
                                        size_t>()) {}
  __device__ Eigen::Vector2i project(const Eigen::Vector3f point) {
    const float img_x =
        kcam_matrix[0][0] * point[0] / point[2] + kcam_matrix[0][2];
    const float img_y =
        kcam_matrix[1][1] * point[1] / point[2] + kcam_matrix[1][2];

    return Eigen::Vector2i(round(img_x), round(img_y));
  }

  const PackedAccessor<float, 2> kcam_matrix;
};

struct RTCamera {
  RTCamera(torch::Tensor rt_matrix)
      : rt_matrix(rt_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits,
                                            size_t>()) {}

  __device__ Eigen::Vector3f transform(const Eigen::Vector3f point) const {
    const auto mtx = rt_matrix;
    const float px = mtx[0][0] * point[0] + mtx[0][1] * point[1] +
                     mtx[0][2] * point[2] + mtx[0][3];
    const float py = mtx[1][0] * point[0] + mtx[1][1] * point[1] +
                     mtx[1][2] * point[2] + mtx[1][3];
    const float pz = mtx[2][0] * point[0] + mtx[2][1] * point[1] +
                     mtx[2][2] * point[2] + mtx[2][3];

    return Eigen::Vector3f(px, py, pz);
  }
  const PackedAccessor<float, 2> rt_matrix;
};

struct JacobianKernel {
  const PackedAccessor<float, 3> points0;
  const PackedAccessor<float, 3> normals0;
  const PackedAccessor<uint8_t, 2> mask0;
  const PackedAccessor<float, 2> points1;
  const PackedAccessor<uint8_t, 1> mask1;
  KCamera kcam;
  RTCamera prev_rt_cam;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;

  JacobianKernel(const PackedAccessor<float, 3> points0,
                 const PackedAccessor<float, 3> normals0,
                 const PackedAccessor<uint8_t, 2> mask0,
                 const PackedAccessor<float, 2> points1,
                 const PackedAccessor<uint8_t, 1> mask1, KCamera kcam,
                 RTCamera prev_rt_cam, PackedAccessor<float, 2> jacobian,
                 PackedAccessor<float, 1> residual)
      : points0(points0),
        normals0(normals0),
        mask0(mask0),
        points1(points1),
        mask1(mask1),
        kcam(kcam),
        prev_rt_cam(prev_rt_cam),
        jacobian(jacobian),
        residual(residual) {}

  __device__ void EstimateJacobian() {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= points1.size(0)) return;

    jacobian[ri][0] = 0.0f;
    jacobian[ri][1] = 0.0f;
    jacobian[ri][2] = 0.0f;
    jacobian[ri][3] = 0.0f;
    jacobian[ri][4] = 0.0f;
    jacobian[ri][5] = 0.0f;
    residual[ri] = 0.0f;

    if (mask1[ri] == 0) return;

    const int width = points0.size(1);
    const int height = points0.size(0);

    const Eigen::Vector3f p1_on_prev =
        prev_rt_cam.transform(to_vec3<float>(points1[ri]));
    Eigen::Vector2i p1_proj = kcam.project(p1_on_prev);
    if (p1_proj[0] < 0 || p1_proj[0] >= width || p1_proj[1] < 0 ||
        p1_proj[1] >= height)
      return;
    if (mask0[p1_proj[1]][p1_proj[0]] == 0) return;
    const Eigen::Vector3f point0(
        to_vec3<float>(points0[p1_proj[1]][p1_proj[0]]));
    const Eigen::Vector3f normal0(
        to_vec3<float>(normals0[p1_proj[1]][p1_proj[0]]));

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

__global__ void EstimateJacobian_gpu_kernel(JacobianKernel kernel) {
  kernel.EstimateJacobian();
}

void EstimateJacobian_gpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor mask0,
                          const torch::Tensor points1,
                          const torch::Tensor mask1, const torch::Tensor kcam,
                          const torch::Tensor rt_cam, torch::Tensor jacobian,
                          torch::Tensor residual) {
  FTB_CHECK(points0.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(normals0.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(mask0.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(points1.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(mask1.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(kcam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(rt_cam.is_cuda(), "Expected a cuda tensor");
  FTB_CHECK(jacobian.is_cuda(), "Expected a cuda tensor");

  FTB_CHECK(residual.is_cuda(), "Expected a cuda tensor");

  JacobianKernel estm_kern(
      points0.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      normals0.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      mask0.packed_accessor<uint8_t, 2, torch::RestrictPtrTraits, size_t>(),
      points1.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      mask1.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      KCamera(kcam), RTCamera(rt_cam),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>());

  CudaKernelDims kl = Get1DKernelDims(points1.size(0));
  EstimateJacobian_gpu_kernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb