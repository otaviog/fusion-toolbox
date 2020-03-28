#include "processing.hpp"

#include "accessor.hpp"
#include "cuda_utils.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace slamtb {
template <Device dev, typename scalar_t>
struct DownsampleXYZNearestKernel {
  const typename Accessor<dev, scalar_t, 3>::T xyz_img;
  const typename Accessor<dev, bool, 2>::T mask;
  scalar_t inv_scale;
  typename Accessor<dev, scalar_t, 3>::T dst;

  DownsampleXYZNearestKernel(const torch::Tensor &xyz_img,
                             const torch::Tensor &mask, scalar_t scale,
                             torch::Tensor dst)
      : xyz_img(Accessor<dev, scalar_t, 3>::Get(xyz_img)),
        mask(Accessor<dev, bool, 2>::Get(mask)),
        inv_scale(scalar_t(1) / scale),
        dst(Accessor<dev, scalar_t, 3>::Get(dst)) {}

  FTB_DEVICE_HOST void operator()(int dst_row, int dst_col) {
    const int center_src_row = dst_row * inv_scale,
              center_src_col = dst_col * inv_scale;

    const int src_width = xyz_img.size(1), src_height = xyz_img.size(0);
    const int where[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    Vector<scalar_t, 3> xyz_mean(0, 0, 0);
    int xyz_count = 0;

    for (int i = 0; i < 4; ++i) {
      const int src_row = center_src_row + where[i][0];
      const int src_col = center_src_col + where[i][1];

      if (src_col < 0 || src_col >= src_width) continue;
      if (!mask[src_row][src_col]) continue;

      const Vector<scalar_t, 3> src_xyz =
          to_vec3<scalar_t>(xyz_img[src_row][src_col]);
      xyz_mean += src_xyz;
      ++xyz_count;
    }

    xyz_mean /= scalar_t(xyz_count);

    Vector<scalar_t, 3> nearest_xyz(0, 0, 0);
    scalar_t best_dist = NumericLimits<dev, scalar_t>::infinity();

    for (int i = 0; i < 4; ++i) {
      const int src_row = center_src_row + where[i][0];
      const int src_col = center_src_col + where[i][1];

      if (src_col < 0 || src_col >= src_width) continue;
      if (!mask[src_row][src_col]) continue;

      const Vector<scalar_t, 3> src_xyz =
          to_vec3<scalar_t>(xyz_img[src_row][src_col]);
      const scalar_t dist = (src_xyz - xyz_mean).squaredNorm();
      if (dist < best_dist) {
        best_dist = dist;
        nearest_xyz = src_xyz;
      }
    }

    dst[dst_row][dst_col][0] = nearest_xyz[0];
    dst[dst_row][dst_col][1] = nearest_xyz[1];
    dst[dst_row][dst_col][2] = nearest_xyz[2];
  }
};

void Processing::DownsampleXYZ(const torch::Tensor &xyz_image,
                               const torch::Tensor &mask, float scale,
                               torch::Tensor dst, bool normalize,
                               DownsampleXYZMethod method) {
  const auto reference_dev = xyz_image.device();
  FTB_CHECK_DEVICE(reference_dev, mask);
  FTB_CHECK_DEVICE(reference_dev, dst);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        xyz_image.scalar_type(), "DownsambleXYZ", ([&] {
          DownsampleXYZNearestKernel<kCUDA, scalar_t> kernel(xyz_image, mask,
                                                             scale, dst);
          Launch2DKernelCUDA(kernel, dst.size(1), dst.size(0));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        xyz_image.scalar_type(), "DownsambleXYZ", ([&] {
          DownsampleXYZNearestKernel<kCPU, scalar_t> kernel(xyz_image, mask,
                                                            scale, dst);
          Launch2DKernelCPU(kernel, dst.size(1), dst.size(0));
        }));
  }
}

template <Device dev>
struct DownsampleMaskKernel {
  const typename Accessor<dev, bool, 2>::T mask;
  const float inv_scale;
  typename Accessor<dev, bool, 2>::T dst;

  DownsampleMaskKernel(const torch::Tensor &mask, float scale,
                       torch::Tensor dst)
      : mask(Accessor<dev, bool, 2>::Get(mask)),
        inv_scale(1.0f / scale),
        dst(Accessor<dev, bool, 2>::Get(dst)) {}

  FTB_DEVICE_HOST void operator()(int dst_row, int dst_col) {
    const int src_row = int(dst_row * inv_scale),
              src_col = int(dst_col * inv_scale);
    dst[dst_row][dst_col] = mask[src_row][src_col];
  }
};

void Processing::DownsampleMask(const torch::Tensor &mask, float scale,
                                torch::Tensor dst) {
  const auto reference_dev = mask.device();
  FTB_CHECK_DEVICE(reference_dev, dst);

  if (reference_dev.is_cuda()) {
    DownsampleMaskKernel<kCUDA> kernel(mask, scale, dst);
    Launch2DKernelCUDA(kernel, dst.size(1), dst.size(0));
  } else {
    DownsampleMaskKernel<kCPU> kernel(mask, scale, dst);
    Launch2DKernelCPU(kernel, dst.size(1), dst.size(0));
  }
}

template <Device dev, typename scalar_t>
struct DownsampleFeatureKernel {
  const typename Accessor<dev, scalar_t, 3>::T feature_img;
  scalar_t inv_scale;
  typename Accessor<dev, scalar_t, 3>::T dst;

  DownsampleFeatureKernel(const torch::Tensor &feature_img, scalar_t scale,
                          torch::Tensor dst)
      : feature_img(Accessor<dev, scalar_t, 3>::Get(feature_img)),
        inv_scale(scalar_t(1) / scale),
        dst(Accessor<dev, scalar_t, 3>::Get(dst)) {}

  FTB_DEVICE_HOST void operator()(int dst_row, int dst_col) {
    const int center_src_row = dst_row * inv_scale,
              center_src_col = dst_col * inv_scale;

    const int src_width = feature_img.size(2), src_height = feature_img.size(1);
    const int where[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    for (int channel = 0; channel < feature_img.size(0); ++channel) {
      scalar_t channel_sum = 0;
      int count = 0;
      for (int i = 0; i < 4; ++i) {
        const int src_row = center_src_row + where[i][0];
        const int src_col = center_src_col + where[i][1];

        if (src_col < 0 || src_col >= src_width) continue;

        const scalar_t value = feature_img[channel][src_row][src_col];
        channel_sum += value;
        ++count;
      }

      dst[channel][dst_row][dst_col] = channel_sum / scalar_t(count);
    }
  }
};

void Processing::DownsampleFeatures(const torch::Tensor &feature_image,
                                    float scale, torch::Tensor dst) {
  const auto reference_dev = feature_image.device();
  FTB_CHECK_DEVICE(reference_dev, dst);

  if (reference_dev.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        feature_image.scalar_type(), "DownsampleFeatures", ([&] {
          DownsampleFeatureKernel<kCUDA, scalar_t> kernel(feature_image, scale,
                                                          dst);
          Launch2DKernelCUDA(kernel, dst.size(2), dst.size(1));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        feature_image.scalar_type(), "DownsampleFeature", ([&] {
          DownsampleFeatureKernel<kCPU, scalar_t> kernel(feature_image, scale,
                                                         dst);
          Launch2DKernelCPU(kernel, dst.size(2), dst.size(1));
        }));
  }
}
}  // namespace slamtb
