#pragma once

#include <string>

#include <torch/torch.h>

#include "error.hpp"

namespace pybind11 {
class module;
}

namespace fiontb {
struct SurfelCloud {
  torch::Tensor positions, confidences, normals, radii, colors, times, features;

  static void RegisterPybind(pybind11::module &m);

  void Allocate(int64_t size, int64_t feature_size, torch::Device device);

  void CheckDevice(const torch::Device &dev) const {
    FTB_CHECK_DEVICE(dev, positions);
    FTB_CHECK_DEVICE(dev, confidences);
    FTB_CHECK_DEVICE(dev, normals);
    FTB_CHECK_DEVICE(dev, radii);
    FTB_CHECK_DEVICE(dev, colors);
    FTB_CHECK_DEVICE(dev, times);
  }

  int64_t get_size() const { return positions.size(0); }

  int64_t get_feature_size() const { return features.size(0); }
};

class SurfelAllocator {
 public:
  SurfelAllocator(int max_surfels);

  static void RegisterPybind(pybind11::module &m);

  void Allocate(torch::Tensor out_free_indices);

  void Free(const torch::Tensor &indices);

  void FreeAll();

  void Copy_(const SurfelAllocator &other);

  int get_max_size() const { return max_surfels_; }

  int get_allocated_size() const {
    return max_surfels_ - int(free_indices_.size());
  }

  int get_free_size() const { return int(free_indices_.size()); }

 private:
  int max_surfels_;
  std::deque<int32_t> free_indices_;
};

struct SurfelOp {
  static void ComputeConfidences(const torch::Tensor &kcam, float weight,
                                 float max_center_distance,
                                 torch::Tensor confidences);

  static void ComputeRadii(const torch::Tensor &kcam,
                           const torch::Tensor &normals, torch::Tensor radii);

  static void Downsample(const SurfelCloud &surfel_cloud, float voxel_size,
                         SurfelCloud &out_surfel_cloud);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
