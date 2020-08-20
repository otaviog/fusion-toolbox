#include "surfel.hpp"

#include "accessor.hpp"
#include "kernel.hpp"
#include "surfel_fusion_common.hpp"

namespace slamtb {
template <Device dev>
struct MergeKernel {
  const SurfelCloudAccessor<dev> self;
  const SurfelCloudAccessor<dev> other;
  const typename Accessor<dev, int32_t, 2>::T index;
  const int32_t N;
  SurfelCloudAccessor<dev> merged;

  MergeKernel(const SurfelCloud &self, const SurfelCloud &other,
              const torch::Tensor index, SurfelCloud merged)
      : self(self),
        other(other),
        index(Accessor<dev, int32_t, 2>::Get(index)),
        N(self.get_size()),
        merged(merged) {}

  STB_DEVICE_HOST void operator()(int idx) {
    Eigen::Vector3f pos = other.point(idx);
    Eigen::Vector3f normal = other.normal(idx);
    Eigen::Vector3f color = other.color(idx);
    float conf = other.confidences[idx];
    float radius = other.radii[idx];
    int32_t time = other.times[idx];

    int count = 1;
    for (int k = 0; k < index.size(1); ++k) {
      const int32_t self_index = index[idx][k];
      if (self_index == N) break;

      pos += self.point(self_index);
      normal += self.normal(self_index);
      color += self.color(self_index);
      conf += self.confidences[self_index];
      radius += self.radii[self_index];
      time += self.times[self_index];
      ++count;
    }

    merged.set_point(idx, pos / count);
    merged.set_normal(idx, normal / count);
    merged.set_color(idx, color / count);
    merged.confidences[idx] = conf / count;
    merged.radii[idx] = radius / count;
    merged.times[idx] = int32_t(float(time) / count);
  }
};

void SurfelOp::Merge(const SurfelCloud &self, const SurfelCloud &other,
                     const torch::Tensor &index, SurfelCloud merged) {
  auto reference_dev = self.points.device();
  other.CheckDevice(reference_dev);
  STB_CHECK_DEVICE(reference_dev, index);
  merged.CheckDevice(reference_dev);

  if (reference_dev.is_cuda()) {
    MergeKernel<kCUDA> kernel(self, other, index, merged);
    Launch1DKernelCUDA(kernel, other.get_size());
  } else {
    MergeKernel<kCPU> kernel(self, other, index, merged);
    Launch1DKernelCPU(kernel, other.get_size());
  }
}

}  // namespace slamtb
