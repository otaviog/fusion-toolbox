#include <cuda_runtime.h>

#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "math.hpp"

// TODO: keep the code from Bad-SLAM?

namespace fiontb {

namespace {
__global__ void ComputeCentralDifferences_gpu_kernel(
    const PackedAccessor<float, 3> xyz_acc,
    const PackedAccessor<uint8_t, 2> mask_acc,
    PackedAccessor<float, 3> normal_acc) {
  const int iwidth = xyz_acc.size(1);
  const int iheight = xyz_acc.size(0);

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= iheight || col >= iwidth) return;

  normal_acc[row][col][0] = normal_acc[row][col][1] = normal_acc[row][col][2] =
      0;

  if (mask_acc[row][col] == 0) return;

  const Eigen::Vector3f center(to_vec3<float>(xyz_acc[row][col]));

  Eigen::Vector3f left = Eigen::Vector3f::Zero();
  if (col > 0 && mask_acc[row][col - 1] == 1) {
    left = Eigen::Vector3f(to_vec3<float>(xyz_acc[row][col - 1]));
  }

  Eigen::Vector3f right = Eigen::Vector3f::Zero();
  if (col < iwidth - 1 && mask_acc[row][col + 1] == 1) {
    right = Eigen::Vector3f(to_vec3<float>(xyz_acc[row][col + 1]));
  }

  Eigen::Vector3f top = Eigen::Vector3f::Zero();
  if (row > 0 && mask_acc[row - 1][col] == 1) {
    top = Eigen::Vector3f(to_vec3<float>(xyz_acc[row - 1][col]));
  }

  Eigen::Vector3f bottom = Eigen::Vector3f::Zero();
  if (row < iheight - 1 && mask_acc[row + 1][col] == 1) {
    bottom = Eigen::Vector3f(to_vec3<float>(xyz_acc[row + 1][col]));
  }
  constexpr float kRatioThreshold = 2.f;
  constexpr float kRatioThresholdSquared = kRatioThreshold * kRatioThreshold;

  float left_dist_squared = (left - center).squaredNorm();
  float right_dist_squared = (right - center).squaredNorm();
  float left_right_ratio = left_dist_squared / right_dist_squared;

  Eigen::Vector3f left_to_right;
  if (left_right_ratio < kRatioThresholdSquared &&
      left_right_ratio > 1.f / kRatioThresholdSquared) {
    left_to_right = right - left;
  } else if (left_dist_squared < right_dist_squared) {
    left_to_right = center - left;
  } else {  // left_dist_squared >= right_dist_squared
    left_to_right = right - center;
  }

  float bottom_dist_squared = (bottom - center).squaredNorm();
  float top_dist_squared = (top - center).squaredNorm();
  float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
  Eigen::Vector3f bottom_to_top;
  if (bottom_top_ratio < kRatioThresholdSquared &&
      bottom_top_ratio > 1.f / kRatioThresholdSquared) {
    bottom_to_top = top - bottom;
  } else if (bottom_dist_squared < top_dist_squared) {
    bottom_to_top = center - bottom;
  } else {  // bottom_dist_squared >= top_dist_squared
    bottom_to_top = top - center;
  }

  Eigen::Vector3f normal = left_to_right.cross(bottom_to_top);
  const float length = normal.norm();
  if (!(length > 1e-6f)) {
    normal = Eigen::Vector3f(0, 0, -1);
  } else {
    normal.normalize();
  }

  const Eigen::Vector3f xvec =
      ((center + left) * 0.5) - ((center + right) * 0.5);
  const Eigen::Vector3f yvec = (center + top) * 0.5 - (center + bottom) * 0.5;

  normal_acc[row][col][0] = normal[0];
  normal_acc[row][col][1] = normal[1];
  normal_acc[row][col][2] = normal[2];
}
}  // namespace

void ComputeCentralDifferences_gpu(const torch::Tensor xyz_image,
                                   const torch::Tensor mask_image,
                                   torch::Tensor normals) {
  const CudaKernelDims kern_dims =
      Get2DKernelDims(xyz_image.size(1), xyz_image.size(0));

  ComputeCentralDifferences_gpu_kernel<<<kern_dims.grid, kern_dims.block>>>(
      xyz_image.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      mask_image
          .packed_accessor<uint8_t, 2, torch::RestrictPtrTraits, size_t>(),
      normals.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}

}  // namespace fiontb