#include "surfel.hpp"
#include "surfel_fusion_common.hpp"

#include <unordered_map>

#include "aabb.hpp"
#include "accessor.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace slamtb {
struct SurfelAccum {
  Eigen::Vector3f point;
  Eigen::Vector3f normal;
  Eigen::Vector3f color;
  torch::Tensor feature;
  bool feature_allocated;

  float confidence;
  float radius;
  int time;
  int count;

  SurfelAccum()
      : point(0, 0, 0),
        normal(0, 0, 0),
        color(0, 0, 0),
        feature_allocated(false),
        confidence(0),
        radius(0),
        time(0),
        count(0) {}

  void AllocateFeature(int feature_size, torch::Device device) {
    if (!feature_allocated) {
      feature = torch::zeros(
          {feature_size}, torch::TensorOptions(device).dtype(torch::kFloat32));
      feature_allocated = true;
    }
  }
};

template <Device dev>
struct SurfelAccumVolume {
  typedef std::unordered_map<int, SurfelAccum> MapType;

  MapType volume;
  Eigen::Vector3f min_point;
  float voxel_size;
  int dims[3];

  SurfelAccumVolume(const AABB &aabb, float voxel_size)
      : min_point(aabb.get_min()), voxel_size(voxel_size) {
    const Eigen::Vector3f dim_dists = aabb.get_max() - aabb.get_min();
    dims[0] = dim_dists[2] / voxel_size;
    dims[1] = dim_dists[1] / voxel_size;
    dims[2] = dim_dists[0] / voxel_size;
  }

  int Hash(float x, float y, float z) {
    const int col = (x - min_point[0]) / voxel_size;
    const int row = (y - min_point[1]) / voxel_size;
    const int depth = (z - min_point[2]) / voxel_size;

    const int voxel_id = depth * dims[0] * dims[1] + row * dims[2] + col;

    return voxel_id;
  }

  SurfelAccum &operator()(float x, float y, float z) {
    return volume[Hash(x, y, z)];
  }

  bool Find(int voxel_id, SurfelAccum *found) const {
    const auto iter = volume.find(voxel_id);
    if (iter == volume.end()) {
      return false;
    }
    *found = iter->second;

    return true;
  }

  torch::Tensor GetVoxelIDs() const {
    torch::Tensor voxel_ids =
        torch::empty({int64_t(volume.size())},
                     torch::TensorOptions(torch::kCPU).dtype(torch::kInt32));
    auto acc = voxel_ids.accessor<int32_t, 1>();
    int count = 0;
    for (auto it = volume.begin(); it != volume.end(); ++it) {
      acc[count++] = it->first;
    }
    return voxel_ids;
  }
};

template <Device dev>
struct AccumKernel {
  const SurfelCloudAccessor<dev> surfels;
  const int64_t feature_size;
  SurfelAccumVolume<dev> &accum_surfel_volume;

  AccumKernel(const SurfelCloud &surfels,
              SurfelAccumVolume<dev> &accum_surfel_volume)
      : surfels(surfels),
        feature_size(surfels.get_feature_size()),
        accum_surfel_volume(accum_surfel_volume) {}

  FTB_DEVICE_HOST void operator()(int idx) {
    const Eigen::Vector3f point = surfels.position(idx);

    SurfelAccum &accum_surfel =
        accum_surfel_volume(point[0], point[1], point[2]);

    accum_surfel.point += point;
    accum_surfel.normal += surfels.normal(idx);
    accum_surfel.color += surfels.color(idx);
    accum_surfel.confidence += surfels.confidences[idx];
    accum_surfel.radius += surfels.radii[idx];

    accum_surfel.AllocateFeature(feature_size, torch::kCPU);
    auto acc = accum_surfel.feature.accessor<float, 1>();
    for (int64_t i = 0; i < feature_size; ++i) {
      acc[i] += surfels.features[i][idx];
    }

    accum_surfel.count += 1;
  }
};

template <Device dev>
struct MergeKernel {
  const SurfelAccumVolume<dev> &accum_volume;
  const typename Accessor<dev, int32_t, 1>::T accum_voxels;
  SurfelCloudAccessor<dev> out_surfels;
  const int64_t feature_size;

  MergeKernel(const SurfelAccumVolume<dev> &accum_volume,
              const torch::Tensor &accum_voxels, const SurfelCloud &out_surfels)
      : accum_volume(accum_volume),
        accum_voxels(Accessor<dev, int32_t, 1>::Get(accum_voxels)),
        out_surfels(out_surfels),
        feature_size(out_surfels.get_feature_size()) {}

  void operator()(int idx) {
    const int voxel_id = accum_voxels[idx];

    SurfelAccum accum;
    accum_volume.Find(voxel_id, &accum);

	const float inv_accum_count = 1.0f / float(accum.count);
    out_surfels.set_position(idx, accum.point * inv_accum_count);
    out_surfels.set_normal(idx, accum.normal * inv_accum_count);
    out_surfels.set_color(idx, accum.color * inv_accum_count);
    out_surfels.confidences[idx] = accum.confidence * inv_accum_count;
    out_surfels.radii[idx] = accum.radius * inv_accum_count;
    out_surfels.times[idx] = accum.time * inv_accum_count;

    const auto acc = accum.feature.accessor<float, 1>();
	
    for (int64_t i = 0; i < feature_size; ++i) {
      out_surfels.features[i][idx] = acc[i] * inv_accum_count;
    }
  }
};

void SurfelOp::Downsample(const SurfelCloud &surfel_cloud, float voxel_size,
                          SurfelCloud &out_surfel_cloud) {
  surfel_cloud.CheckDevice(torch::kCPU);
  
  const AABB cloud_box(surfel_cloud.positions);
  SurfelAccumVolume<kCPU> accum_volume(cloud_box, voxel_size);

  AccumKernel<kCPU> accum_kernel(surfel_cloud, accum_volume);
  Launch1DKernelCPU(accum_kernel, surfel_cloud.get_size(), true);

  torch::Tensor voxel_ids = accum_volume.GetVoxelIDs();
  out_surfel_cloud.Allocate(voxel_ids.size(0), surfel_cloud.get_feature_size(),
                            torch::kCPU);

  MergeKernel<kCPU> merge_kernel(accum_volume, voxel_ids, out_surfel_cloud);
  Launch1DKernelCPU(merge_kernel, voxel_ids.size(0));
}


};  // namespace slamtb
