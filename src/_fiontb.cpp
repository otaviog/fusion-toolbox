#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "normals.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_fiontb, m) {
  using namespace fiontb;
  m.def("calculate_depth_normals", &CalculateFrameNormals);
}
