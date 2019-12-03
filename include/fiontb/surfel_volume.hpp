#pragma once

#include "hash_volume.hpp"
#include "surfel.hpp"

namespace fiontb {
class SurfelVolume {
 public:
  struct Surfel {
    Eigen::Vector3f point, normal, color;
    float confidence, radius;
    int32_t time, count;
    torch::Tensor feature;
    bool feature_allocated;
    Surfel()
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
        feature =
            torch::zeros({feature_size},
                         torch::TensorOptions(device).dtype(torch::kFloat32));
        feature_allocated = true;
      }
    }
  };

  SurfelVolume() {
  }
  
  void Merge(const SurfelCloud &surfels);

  void ToCloud(SurfelCloud out);

  static void RegisterPybind(pybind11::module &m);

 private:
  HashVolume<Surfel> volume_;
};

}  // namespace fiontb
