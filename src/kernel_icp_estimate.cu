#include "icpodometry.hpp"

#include <sophus/se3.hpp>

#include "cuda_utils.hpp"
#include "math.hpp"

namespace fiontb {
inline __device__ void project_pinhole(const PackedAccessor<float, 2> kcam,
                                       Eigen::Vector3f point, int *x, int *y) {
  const float img_x = (kcam[0][0] * point[0] + kcam[0][2]) / point[2];
  const float img_y = (kcam[1][1] * point[1] + kcam[1][2]) / point[2];

  *x = round(img_x);
  *y = round(img_y);
}

struct JacobianKernel {
  const PackedAccessor<float, 3> points0;
  const PackedAccessor<float, 3> normals0;
  const PackedAccessor<float, 2> points1;
  const PackedAccessor<float, 2> kcam_matrix;
  const PackedAccessor<float, 1> params;

  PackedAccessor<float, 2> jacobian;
  PackedAccessor<float, 1> residual;
  float epsilon;

  JacobianKernel(const PackedAccessor<float, 3> points0,
                 const PackedAccessor<float, 3> normals0,
                 const PackedAccessor<float, 2> points1,
                 const PackedAccessor<float, 2> kcam_matrix,
                 const PackedAccessor<float, 1> params,
                 PackedAccessor<float, 2> jacobian,
                 PackedAccessor<float, 1> residual, const float epsilon)
      : points0(points0),
        normals0(normals0),
        points1(points1),
        kcam_matrix(kcam_matrix),
        params(params),
        jacobian(jacobian),
        residual(residual),
        epsilon(epsilon) {}

  __device__ void EstimateJacobian() {
    const int ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri > points1.size(0)) return;

    for (int pj=0; pj<7; ++pj) {
      Eigen::Matrix<float, 7, 1> raw_params;
      raw_params << params[0], params[1], params[2], params[3], params[4],
          params[5], params[6];
      raw_params[pj] += epsilon;

      Eigen::Map<Sophus::SE3f> model(raw_params.data());
      Eigen::Vector3f point = (model * to_vec3<float>(points1[ri]));
      int img_x, img_y;
      project_pinhole(kcam_matrix, point, &img_x, &img_y);

      if (img_x < 0 || img_x >= points0.size(1) || img_y < 0 ||
          img_y >= points0.size(0))
      return;

      Eigen::Vector3f point0(to_vec3<float>(points0[img_y][img_x]));
      Eigen::Vector3f normal0(to_vec3<float>(normals0[img_y][img_x]));

      const float curr_residual = (point0 - point).dot(normal0);
      float params_length = 0.0f;
      for (int i = 0; i < 7; ++i) params_length += raw_params[i] * raw_params[i];

      jacobian[ri][pj] = (prev_residual[ri] - curr_residual) / params_length;
    }
  }
};

__global__ void EstimateJacobian_gpu_kernel(JacobianKernel kernel) {
  kernel.EstimateJacobian();
}

void EstimateJacobian_gpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor points1, const torch::Tensor kcam,
                          const torch::Tensor params, torch::Tensor jacobian,
                          torch::Tensor residual) {
  JacobianKernel estm_kern(
      points0.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      normals0.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      points1.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      kcam.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      params.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
      jacobian.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      residual.packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
      0.001f);

  CudaKernelDims kl = Get2DKernelDims(points1.size(0), 6);
  EstimateJacobian_gpu_kernel<<<kl.grid, kl.block>>>(estm_kern);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

#if 0
static torch::Tensor estimate_update(torch::Tensor points0, torch::Tensor normals0,
                                torch::Tensor points1, torch::Tensor kcam,
                                torch::Tensor transform,
                                torch::Tensor curr_params,
                                torch::Tensor curr_residual,
                                torch::Tensor prev_residual) {

  auto jacobian = EstimateJacobian_gpu(points0, normals0, points1, kcam, transform, curr_params,
                                       curr_residual, prev_residual);
  torch::Tensor JtJ = jacobian.transpose(1, 0).matmul(jacobian);
  auto inv_JtJ = JtJ.inverse();
  auto Jr = jacobian.matmul(curr_residual);

  torch::Tensor param_update = inv_JtJ.matmul(Jr);
  return param_update;
}
#endif

ICPOdometry::ICPOdometry(std::vector<float> scales, std::vector<int> num_iters)
    : scales_(scales), num_iters_(num_iters) {}

torch::Tensor ICPOdometry::Estimate(torch::Tensor points0,
                                    torch::Tensor normals0,
                                    torch::Tensor points1, torch::Tensor kcam,
                                    torch::Tensor transform) {
  torch::Tensor params = torch::zeros(
      {6}, torch::TensorOptions(torch::kFloat).device(torch::kCUDA, 0));
#if 0
  torch::Tensor residual0 =
      torch::zeros({points1.size(0)},
                   torch::TensorOptions(torch::kFloat).device(torch::kCUDA, 0));
  torch::Tensor residual1 = residual0.clone();

  for (int scale_idx = 0; scale_idx < scales_.size(); ++scale_idx) {
    const float scale = scales_[scale_idx];  // TODO: rescale image
    const int num_iters = num_iters_[scale_idx];

    for (int i = 0; i < num_iters; ++i) {
      torch::Tensor param_update =
          estimate_update(points0, normals0, points1, kcam, transform, params,
                          residual0, residual1);
      params = params + param_update;
      std::swap(residual0, residual1);
    }
  }
#endif
  return params;
}
}  // namespace fiontb