#pragma once

#include <torch/torch.h>
#include <string>

namespace pybind11 {
class module;
}

namespace fiontb {
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

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace fiontb
