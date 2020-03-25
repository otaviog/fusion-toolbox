#include "se3.hpp"

#include <torch/csrc/utils/pybind.h>

namespace slamtb {
void ExpRtToMatrixOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<ExpRtToMatrixOp>(m, "ExpRtToMatrixOp")
      .def_static("forward", &ExpRtToMatrixOp::Forward)
      .def_static("backward", &ExpRtToMatrixOp::Backward);
}

void MatrixToExpRtOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<MatrixToExpRtOp>(m, "MatrixToExpRtOp")
      .def_static("forward", &MatrixToExpRtOp::Forward);
}

}  // namespace slamtb
