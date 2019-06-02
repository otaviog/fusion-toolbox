#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "dense_volume.hpp"
#include "filtering.hpp"
#include "indexmap.hpp"
#include "normals.hpp"
#include "octree.hpp"
#include "sparse_volume.hpp"
#include "surfel_fusion.hpp"
#include "tsdf_fusion.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_fiontb, m) {
  using namespace fiontb;
  m.def("calculate_depth_normals", &CalculateFrameNormals);
  m.def("filter_search", &FilterSearch);
  m.def("filter_depth_image", &FilterDepthImage);

  py::class_<IndexMap>(m, "IndexMap")
      .def(py::init<torch::Tensor, torch::Tensor, int, int, int, int>())
      .def("query", &IndexMap::Query)
      .def_readwrite("grid", &IndexMap::grid_)
      .def_readwrite("model", &IndexMap::model_points_);

  py::class_<Octree>(m, "Octree")
      .def(py::init<torch::Tensor, int>())
      .def("query", &Octree::Query);

  py::class_<DenseVolume, shared_ptr<DenseVolume>>(m, "DenseVolume")
      .def(py::init<int, float, Eigen::Vector3i>())
      .def("to_point_cloud", &DenseVolume::ToPointCloud)
      .def_readwrite("sdf", &DenseVolume::sdf)
      .def_readwrite("weights", &DenseVolume::weights);

  py::class_<SparseVolume, shared_ptr<SparseVolume>>(m, "SparseVolume")
      .def(py::init<float, int>())
      .def("get_unit", &SparseVolume::GetUnit);

  m.def("fuse_dense_volume", &FuseDenseVolume);
  m.def("fuse_sparse_volume", &FuseSparseVolume);
}
