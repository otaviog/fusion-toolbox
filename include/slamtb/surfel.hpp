#pragma once

#include <string>

#include <torch/torch.h>

#include "error.hpp"

namespace pybind11 {
class module;
}

namespace slamtb {
struct SurfelCloud {
  torch::Tensor points, confidences, normals, radii, colors, times, features;

  static void RegisterPybind(pybind11::module &m);

  void Allocate(int64_t size, int64_t feature_size, torch::Device device);

  void CheckDevice(const torch::Device &dev) const {
    FTB_CHECK_DEVICE(dev, points);
    FTB_CHECK_DEVICE(dev, confidences);
    FTB_CHECK_DEVICE(dev, normals);
    FTB_CHECK_DEVICE(dev, radii);
    FTB_CHECK_DEVICE(dev, colors);
    FTB_CHECK_DEVICE(dev, times);
  }

  int64_t get_size() const { return points.size(0); }

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
  /**
   * Compute confidences by its distance from the center.
   *
   * @param kcam A (2x3) intrinsic matrix.
   * @param weight Confidence scaling.
   * @param max_center_distance Maximum distance from the center
   * @param confidences Output confidences (N).
   */
  static void ComputeConfidences(const torch::Tensor &kcam, float weight,
                                 float max_center_distance,
                                 torch::Tensor confidences);

  /**
   * Compute Surfel Radii using the formula from Weise, Thibaut,
   * Thomas Wismer, Bastian Leibe, and Luc Van Gool. "In-hand scanning
   * with online loop closure." In 2009 IEEE 12th International
   * Conference on Computer Vision Workshops, ICCV Workshops,
   * pp. 1630-1637. IEEE, 2009.
   *
   */
  static void ComputeRadii(const torch::Tensor &kcam,
                           const torch::Tensor &depths,
                           const torch::Tensor &normals_z, torch::Tensor radii);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace slamtb
