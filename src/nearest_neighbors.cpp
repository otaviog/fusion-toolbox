#include "nearest_neighbors.hpp"

#include <torch/csrc/utils/pybind.h>

namespace slamtb {
void NearestNeighborsOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<NearestNeighborsOp>(m, "NearestNeighborsOp")
      .def_static("forward", &NearestNeighborsOp::Forward)
      .def_static("compute_epsilon_distances",
                  &NearestNeighborsOp::ComputeEpsilonDistances)
      .def_static("backward", &NearestNeighborsOp::Backward);
}
}  // namespace slamtb
