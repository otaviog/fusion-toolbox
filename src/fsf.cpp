#include "fsf.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

static void FSFOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<FSFOp>(m, "FSFOp").def_static("merge", FSFOp::Merge);
}
