#include "surfel.hpp"

#include "accessor.hpp"
#include "camera.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {
template <Device dev, typename scalar_t>
struct ComputeConfidencesKernel {
  const KCamera<dev, scalar_t> kcam;
  const scalar_t constant_weight;
  const scalar_t weight;
  const scalar_t max_center_distance;
  typename Accessor<dev, scalar_t, 2>::T confidences;

  ComputeConfidencesKernel(const torch::Tensor &kcam, scalar_t weight,
                           scalar_t max_center_distance,
                           torch::Tensor confidences)
      : kcam(kcam),
        constant_weight(2.0 * pow(0.6, 2)),
        weight(weight),
        max_center_distance(max_center_distance),
        confidences(Accessor<dev, scalar_t, 2>::Get(confidences)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    const Vector<scalar_t, 2> camera_center(kcam.get_center());
    scalar_t confidence =
        (Vector<scalar_t, 2>(col, row) - camera_center).norm();
    confidence = confidence / max_center_distance;
    confidence = exp(-(confidence * confidence) / constant_weight) * weight;

    confidences[row][col] = confidence;
  }
};

}  // namespace

void SurfelOp::ComputeConfidences(const torch::Tensor &kcam, float weight,
                                  float max_center_distance,
                                  torch::Tensor confidences) {
  const auto ref_device = confidences.device();
  FTB_CHECK_DEVICE(ref_device, kcam);

  const auto ref_type = confidences.scalar_type();
  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "ComputeConfidences", [&] {
      ComputeConfidencesKernel<kCUDA, scalar_t> kernel(
          kcam, weight, max_center_distance, confidences);
      Launch2DKernelCUDA(kernel, confidences.size(1), confidences.size(0));
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "ComputeConfidences", [&] {
      ComputeConfidencesKernel<kCPU, scalar_t> kernel(
          kcam, weight, max_center_distance, confidences);
      Launch2DKernelCPU(kernel, confidences.size(1), confidences.size(0));
    });
  }
}

namespace {
template <Device dev, typename scalar_t>
struct ComputeRadiiKernel {
  const typename Accessor<dev, scalar_t, 1>::T depths;
  const typename Accessor<dev, scalar_t, 1>::T normals_z;
  typename Accessor<dev, scalar_t, 1>::T radii;
  scalar_t focal_len;

  ComputeRadiiKernel(const torch::Tensor &kcam_, const torch::Tensor &depths,
                     const torch::Tensor &normals_z, torch::Tensor radii)
      : depths(Accessor<dev, scalar_t, 1>::Get(depths)),
        normals_z(Accessor<dev, scalar_t, 1>::Get(normals_z)),
        radii(Accessor<dev, scalar_t, 1>::Get(radii)) {
    const KCamera<kCPU, scalar_t> kcam(kcam_);
    focal_len =
        (abs(kcam.matrix[0][0]) + abs(kcam.matrix[1][1])) * 0.5;
  }

  FTB_DEVICE_HOST void operator()(int idx) {
    const scalar_t _1_sqrt_2 = 0.7071067811865475;
    const scalar_t radius = _1_sqrt_2 * (depths[idx] / focal_len);

    radii[idx] = min(radius / abs(normals_z[idx]), 2 * radius);
  }
};
}  // namespace

void SurfelOp::ComputeRadii(const torch::Tensor &kcam,
                            const torch::Tensor &depths,
                            const torch::Tensor &normals_z,
                            torch::Tensor radii) {
  const auto ref_device = normals_z.device();
  FTB_CHECK_DEVICE(ref_device, kcam);
  FTB_CHECK_DEVICE(ref_device, radii);

  const auto ref_type = normals_z.scalar_type();
  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "ComputeRadii", [&] {
      ComputeRadiiKernel<kCUDA, scalar_t> kernel(kcam, depths, normals_z,
                                                 radii);
      Launch1DKernelCUDA(kernel, normals_z.size(0));
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "ComputeRadii", [&] {
      ComputeRadiiKernel<kCPU, scalar_t> kernel(kcam, depths, normals_z, radii);
      Launch1DKernelCPU(kernel, normals_z.size(0));
    });
  }
}

}  // namespace fiontb
