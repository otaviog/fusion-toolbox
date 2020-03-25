#pragma once

#include "hash_volume.hpp"
#include "surfel.hpp"

namespace slamtb {
class SurfelVolume {
 public:
  struct Surfel {
    Eigen::Vector3f point, normal, color;
    float confidence, radius;
    int32_t time, count;
    torch::Tensor feature;
    Surfel()
        : point(0, 0, 0),
          normal(0, 0, 0),
          color(0, 0, 0),
          confidence(0),
          radius(0),
          time(0),
          count(0) {}

    void AllocateFeature(int feature_size, torch::Device device) {
      if (feature.use_count() == 0) {
        feature =
            torch::zeros({feature_size},
                         torch::TensorOptions(device).dtype(torch::kFloat32));
      }
    }

    inline bool has_feature() const {
      return feature.use_count() > 0;
    }
  };

  SurfelVolume(const Eigen::Vector3f &min_point,
               const Eigen::Vector3f &max_point, float voxel_size,
               int feature_size=-1)
      : volume_(min_point, max_point, voxel_size), feature_size_(feature_size) {}

  void Merge(const SurfelCloud &surfels);

  void ToSurfelCloud(SurfelCloud &out) const;

  static void RegisterPybind(pybind11::module &m);

 private:
  HashVolume<Surfel> volume_;
  int feature_size_;
};

}  // namespace slamtb
