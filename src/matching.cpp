#include "matching.hpp"

#include <torch/csrc/utils/pybind.h>

namespace fiontb {
void FPCLMatcherOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<FPCLMatcherOp>(m, "FPCLMatcherOp")
      .def_static("forward", &FPCLMatcherOp::Forward)
      .def_static("backward", &FPCLMatcherOp::Backward);
}
}  // namespace fiontb
