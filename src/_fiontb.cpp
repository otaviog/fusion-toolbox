#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "indexmap.hpp"
#include "normals.hpp"
#include "octree.hpp"
#include "surfel_fusion.hpp"
#include "filtering.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_fiontb, m) {
  using namespace fiontb;
  m.def("calculate_depth_normals", &CalculateFrameNormals);
  m.def("filter_search", &FilterSearch);
  m.def("filter_depth_image", &FilterDepthImage);

  py::class_<IndexMap>(m, "IndexMap")
      .def(py::init<torch::Tensor, int, int>())
      .def("query", &IndexMap::Query)
      .def_readwrite("grid", &IndexMap::grid_)
      .def_readwrite("model", &IndexMap::model_points_);

  py::class_<Octree>(m, "Octree")
      .def(py::init<torch::Tensor, int>())
      .def("query", &Octree::Query);
}
