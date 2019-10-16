#include "surfel.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

namespace fiontb {
void SurfelOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<SurfelOp>(m, "SurfelOp")
      .def_static("compute_confidences", &SurfelOp::ComputeConfidences)
      .def_static("compute_radii", &SurfelOp::ComputeRadii);
}
}  // namespace fiontb
