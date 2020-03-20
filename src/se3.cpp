#include "se3.hpp"

#include <torch/csrc/utils/pybind.h>

namespace fiontb {
void ExpRtToMatrixOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<ExpRtToMatrixOp>(m, "ExpRtToMatrixOp")
      .def_static("forward", &ExpRtToMatrixOp::Forward)
      .def_static("backward", &ExpRtToMatrixOp::Backward);
}

void ExpRtTransformOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<ExpRtTransformOp>(m, "ExpRtTransformOp")
      .def_static("forward", &ExpRtTransformOp::Forward)
      .def_static("backward", &ExpRtTransformOp::Backward);
}

void QuatRtTransformOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<QuatRtTransformOp>(m, "QuatRtTransformOp")
      .def_static("forward", &QuatRtTransformOp::Forward)
      .def_static("backward", &QuatRtTransformOp::Backward);
}
}  // namespace fiontb
