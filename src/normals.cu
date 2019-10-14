#include "normals.hpp"

#include <cuda_runtime.h>

#include "accessor.hpp"
#include "cuda_utils.hpp"
#include "eigen_common.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {

namespace {
template <Device dev, typename scalar_t>
struct CentralDifferencesKernel {
  // TODO: Code taken from Bad-SLAM? Verify the license before open-source
  const typename Accessor<dev, scalar_t, 3>::T xyz;
  const typename Accessor<dev, bool, 2>::T mask;
  typename Accessor<dev, scalar_t, 3>::T out_normal;

  CentralDifferencesKernel(const torch::Tensor &xyz, const torch::Tensor &mask,
                           torch::Tensor out_normal)
      : xyz(Accessor<dev, scalar_t, 3>::Get(xyz)),
        mask(Accessor<dev, bool, 2>::Get(mask)),
        out_normal(Accessor<dev, scalar_t, 3>::Get(out_normal)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    const int iwidth = xyz.size(1);
    const int iheight = xyz.size(0);

    out_normal[row][col][0] = out_normal[row][col][1] = 
        out_normal[row][col][2] = 0;

    if (!mask[row][col]) return;

    const Vector<scalar_t, 3> center(to_vec3<scalar_t>(xyz[row][col]));

    Vector<scalar_t, 3> left = Vector<scalar_t, 3>::Zero();
    if (col > 0 && mask[row][col - 1]) {
      left = Vector<scalar_t, 3>(to_vec3<scalar_t>(xyz[row][col - 1]));
    }

    Vector<scalar_t, 3> right = Vector<scalar_t, 3>::Zero();
    if (col < iwidth - 1 && mask[row][col + 1]) {
      right = Vector<scalar_t, 3>(to_vec3<scalar_t>(xyz[row][col + 1]));
    }

    Vector<scalar_t, 3> top = Vector<scalar_t, 3>::Zero();
    if (row > 0 && mask[row - 1][col]) {
      top = Vector<scalar_t, 3>(to_vec3<scalar_t>(xyz[row - 1][col]));
    }

    Vector<scalar_t, 3> bottom = Vector<scalar_t, 3>::Zero();
    if (row < iheight - 1 && mask[row + 1][col]) {
      bottom = Vector<scalar_t, 3>(to_vec3<scalar_t>(xyz[row + 1][col]));
    }
    constexpr scalar_t kRatioThreshold = 2.f;
    constexpr scalar_t kRatioThresholdSquared =
        kRatioThreshold * kRatioThreshold;

    scalar_t left_dist_squared = (left - center).squaredNorm();
    scalar_t right_dist_squared = (right - center).squaredNorm();
    scalar_t left_right_ratio = left_dist_squared / right_dist_squared;

    Vector<scalar_t, 3> left_to_right;
    if (left_right_ratio < kRatioThresholdSquared &&
        left_right_ratio > 1.f / kRatioThresholdSquared) {
      left_to_right = right - left;
    } else if (left_dist_squared < right_dist_squared) {
      left_to_right = center - left;
    } else {  // left_dist_squared >= right_dist_squared
      left_to_right = right - center;
    }

    scalar_t bottom_dist_squared = (bottom - center).squaredNorm();
    scalar_t top_dist_squared = (top - center).squaredNorm();
    scalar_t bottom_top_ratio = bottom_dist_squared / top_dist_squared;
    Vector<scalar_t, 3> bottom_to_top;
    if (bottom_top_ratio < kRatioThresholdSquared &&
        bottom_top_ratio > 1.f / kRatioThresholdSquared) {
      bottom_to_top = top - bottom;
    } else if (bottom_dist_squared < top_dist_squared) {
      bottom_to_top = center - bottom;
    } else {  // bottom_dist_squared >= top_dist_squared
      bottom_to_top = top - center;
    }

    Vector<scalar_t, 3> normal = left_to_right.cross(bottom_to_top);
    const scalar_t length = normal.norm();
    if (!(length > 1e-6f)) {
      normal = Vector<scalar_t, 3>(0, 0, -1);
    } else {
      normal.normalize();
    }

    const Vector<scalar_t, 3> xvec =
        ((center + left) * 0.5) - ((center + right) * 0.5);
    const Vector<scalar_t, 3> yvec =
        (center + top) * 0.5 - (center + bottom) * 0.5;

    out_normal[row][col][0] = normal[0];
    out_normal[row][col][1] = normal[1];
    out_normal[row][col][2] = normal[2];
  }
};

template <Device dev, typename scalar_t>
struct Average8Kernel {
  const typename Accessor<dev, scalar_t, 3>::T xyz;
  const typename Accessor<dev, bool, 2>::T mask;
  typename Accessor<dev, scalar_t, 3>::T out_normal;

  Average8Kernel(const torch::Tensor &xyz, const torch::Tensor mask,
                 torch::Tensor out_normal)
      : xyz(Accessor<dev, scalar_t, 3>::Get(xyz)),
        mask(Accessor<dev, bool, 2>::Get(mask)),
        out_normal(Accessor<dev, scalar_t, 3>::Get(out_normal)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    out_normal[row][col][0] = out_normal[row][col][1] =
        out_normal[row][col][2] = 0.0f;

    if (!mask[row][col]) return;

    const int iwidth = xyz.size(1);
    const int iheight = xyz.size(0);

    static const int where[][2] = {{0, 1},  {1, 1},   {1, 0},  {1, -1},
                                   {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

    Vector<scalar_t, 3> center(to_vec3<scalar_t>(xyz[row][col]));
    Vector<scalar_t, 3> norm(0, 0, 0);
    int count = 0;

    for (int i = 0; i < 8; ++i) {
      const int r1 = row + where[i][1];
      if (r1 < 0 || r1 >= iheight) continue;

      const int c1 = col + where[i][0];
      if (c1 < 0 || c1 >= iwidth) continue;

      if (!mask[r1][c1]) continue;

      const int j = (i + 2) % 8;
      const int r2 = row + where[j][1];
      if (r2 < 0 || r2 >= iheight) continue;

      const int c2 = col + where[j][0];
      if (c2 < 0 || c2 >= iwidth) continue;

      if (!mask[r2][c2]) continue;

      norm += GetNormal(center, to_vec3<scalar_t>(xyz[r2][c2]),
                        to_vec3<scalar_t>(xyz[r1][c1]));
      ++count;
    }
    if (count > 0) {
      norm = norm / count;
      norm.normalize();
      out_normal[row][col][0] = norm[0];
      out_normal[row][col][1] = norm[1];
      out_normal[row][col][2] = norm[2];
    }
  }
};

template <Device dev, typename scalar_t>
void _EstimateNormals(const torch::Tensor xyz_image,
                      const torch::Tensor mask_image, torch::Tensor out_normals,
                      EstimateNormalsMethod method) {
  switch (method) {
    case kCentralDifferences: {
      CentralDifferencesKernel<dev, scalar_t> kernel(xyz_image, mask_image,
                                                     out_normals);
      if (dev == kCUDA) {
        Launch2DKernelCUDA(kernel, xyz_image.size(1), xyz_image.size(0));
      } else {
        Launch2DKernelCPU(kernel, xyz_image.size(1), xyz_image.size(0));
      }
    } break;
    case kAverage8: {
      Average8Kernel<dev, scalar_t> kernel(xyz_image, mask_image, out_normals);
      if (dev == kCUDA) {
        Launch2DKernelCUDA(kernel, xyz_image.size(1), xyz_image.size(0));
      } else {
        Launch2DKernelCPU(kernel, xyz_image.size(1), xyz_image.size(0));
      }
    } break;
  };
}

}  // namespace

void EstimateNormals(const torch::Tensor xyz_image,
                     const torch::Tensor mask_image, torch::Tensor out_normals,
                     EstimateNormalsMethod method) {
  FTB_CHECK_DEVICE(xyz_image.device(), mask_image);
  FTB_CHECK_DEVICE(xyz_image.device(), out_normals);
  
  if (xyz_image.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(xyz_image.scalar_type(), "EstimateNormals", [&] {
      _EstimateNormals<kCUDA, scalar_t>(xyz_image, mask_image, out_normals,
                                        method);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(xyz_image.scalar_type(), "EstimateNormals", [&] {
      _EstimateNormals<kCPU, scalar_t>(xyz_image, mask_image, out_normals,
                                       method);
    });
  }
}
}  // namespace fiontb