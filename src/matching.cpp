#include "matching.hpp"

#include <torch/csrc/utils/pybind.h>

namespace slamtb {
void FPCLMatcherOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<FPCLMatcherOp>(m, "FPCLMatcherOp")
      .def_static("forward", &FPCLMatcherOp::Forward)
      .def_static("backward", &FPCLMatcherOp::Backward);
}
}  // namespace slamtb
