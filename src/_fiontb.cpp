#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "normals.hpp"
#include "octtree.hpp"
#include "surfel_fusion.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_fiontb, m) {
  using namespace fiontb;
  m.def("calculate_depth_normals", &CalculateFrameNormals);
  m.def("filter_search", &FilterSearch);

  py::class_<IndexMap>(m, "IndexMap")
      .def(py::init<int, int, torch::Tensor>())
      .def("query", &IndexMap::Query)
      .def_readwrite("grid", &IndexMap::grid_)
      .def_readwrite("model", &IndexMap::model_points_);

  py::class_<OctTree>(m, "OctTree")
      .def(py::init<torch::Tensor, int>())
      .def("query", &OctTree::Query);
}
