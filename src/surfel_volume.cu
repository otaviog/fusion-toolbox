#include "surfel_volume.hpp"

#include "aabb.hpp"
#include "accessor.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "surfel_fusion_common.hpp"

namespace slamtb {

namespace {
struct AccumKernel {
  const SurfelCloudAccessor<kCPU> surfels;
  const int64_t feature_size;
  HashVolume<SurfelVolume::Surfel> &volume;

  AccumKernel(const SurfelCloud &surfels,
              HashVolume<SurfelVolume::Surfel> &volume)
      : surfels(surfels),
        feature_size(surfels.get_feature_size()),
        volume(volume) {}

  void operator()(int idx) {
    const Eigen::Vector3f point = surfels.point(idx);
    SurfelVolume::Surfel &surfel = volume(point);

    surfel.point += point;
    surfel.normal += surfels.normal(idx);
    surfel.color += surfels.color(idx);
    surfel.confidence += surfels.confidences[idx];
    surfel.radius += surfels.radii[idx];

    surfel.AllocateFeature(feature_size, torch::kCPU);
    auto acc = surfel.feature.accessor<float, 1>();
    for (int64_t i = 0; i < feature_size; ++i) {
      acc[i] += surfels.features[i][idx];
    }

    surfel.count += 1;
  }
};

struct AverageKernel {
  HashVolume<SurfelVolume::Surfel> &volume;
  const typename Accessor<kCPU, int32_t, 1>::T voxel_ids;

  AverageKernel(HashVolume<SurfelVolume::Surfel> &volume,
                const torch::Tensor &voxel_ids)
      : volume(volume), voxel_ids(Accessor<kCPU, int32_t, 1>::Get(voxel_ids)) {}

  void operator()(int idx) {
    const int voxel_id = voxel_ids[idx];

    SurfelVolume::Surfel &surfel = volume[voxel_id];
    const float inv_count = 1.0f / float(surfel.count);
    surfel.point *= inv_count;
    surfel.normal *= inv_count;
    surfel.color *= inv_count;
    surfel.confidence *= inv_count;
    surfel.radius *= inv_count;
    surfel.time = int(float(surfel.time) * inv_count);
    surfel.count = 1;

    if (surfel.has_feature()) {
      auto acc = surfel.feature.accessor<float, 1>();
      for (int64_t i = 0; i < surfel.feature.size(0); ++i) {
        acc[i] *= inv_count;
      }
    }
  }
};
}  // namespace

void SurfelVolume::Merge(const SurfelCloud &surfels) {
  AccumKernel accum_kernel(surfels, volume_);
  Launch1DKernelCPU(accum_kernel, surfels.get_size(), true);

  torch::Tensor voxel_ids = volume_.GetVoxelIDs();
  AverageKernel merge_kernel(volume_, voxel_ids);
  Launch1DKernelCPU(merge_kernel, voxel_ids.size(0), true);
}

namespace {
struct ToCloudKernel {
  const HashVolume<SurfelVolume::Surfel> &volume;
  const typename Accessor<kCPU, int32_t, 1>::T voxel_ids;
  SurfelCloudAccessor<kCPU> out_surfels;
  const int feature_size;

  ToCloudKernel(const HashVolume<SurfelVolume::Surfel> &volume,
                const torch::Tensor &voxel_ids, SurfelCloud &out_surfels)
      : volume(volume),
        voxel_ids(Accessor<kCPU, int32_t, 1>::Get(voxel_ids)),
        out_surfels(out_surfels),
        feature_size(out_surfels.get_feature_size()) {}

  void operator()(int idx) {
    const int voxel_id = voxel_ids[idx];

    HashVolume<SurfelVolume::Surfel>::const_iterator found =
        volume.FindId(voxel_id);
    if (found == volume.end()) {
      return;
    }
    const SurfelVolume::Surfel accum = found->second;
    const float inv_count = 1.0f / float(accum.count);
    out_surfels.set_point(idx, accum.point * inv_count);
    out_surfels.set_normal(idx, accum.normal * inv_count);
    out_surfels.set_color(idx, accum.color * inv_count);
    out_surfels.confidences[idx] = accum.confidence * inv_count;
    out_surfels.radii[idx] = accum.radius * inv_count;
    out_surfels.times[idx] = int(float(accum.time) * inv_count);

    const auto acc = accum.feature.accessor<float, 1>();
    for (int64_t i = 0; i < feature_size; ++i) {
      out_surfels.features[i][idx] = acc[i]*inv_count;
    }
  }
};
}  // namespace

void SurfelVolume::ToSurfelCloud(SurfelCloud &out) const {
  torch::Tensor voxel_ids = volume_.GetVoxelIDs();
  out.Allocate(voxel_ids.size(0), feature_size_, torch::kCPU);

  ToCloudKernel to_cloud_kernel(volume_, voxel_ids, out);
  Launch1DKernelCPU(to_cloud_kernel, voxel_ids.size(0));
}
}  // namespace slamtb
