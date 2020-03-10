#include "icp_jacobian.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

namespace fiontb {
void ICPJacobian::RegisterPybind(pybind11::module &m) {
  py::class_<ICPJacobian>(m, "ICPJacobian")
      .def_static("estimate_geometric", &ICPJacobian::EstimateGeometric)
      .def_static("estimate_feature", &ICPJacobian::EstimateFeature)
      .def_static("estimate_feature_so3", &ICPJacobian::EstimateFeatureSO3);
}
}
