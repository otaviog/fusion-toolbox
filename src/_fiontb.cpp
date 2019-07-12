#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "dense_volume.hpp"
#include "filtering.hpp"
#include "indexmap.hpp"
#include "metrics.hpp"
#include "normals.hpp"
#include "sparse_volume.hpp"
#include "surfel_fusion.hpp"
#include "trigoctree.hpp"
#include "tsdf_fusion.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_cfiontb, m) {
  using namespace fiontb;

  m.def("estimate_normals", &EstimateNormals);
  py::enum_<EstimateNormalsMethod>(m, "EstimateNormalsMethod")
      .value("CentralDifferences", EstimateNormalsMethod::kCentralDifferences)
      .value("Average8", EstimateNormalsMethod::kAverage8);

  m.def("bilateral_filter_depth_image", &BilateralFilterDepthImage);

  py::class_<TrigOctree>(m, "TrigOctree")
      .def(py::init<torch::Tensor, torch::Tensor, int>())
      .def("query_closest_points", &TrigOctree::QueryClosest);

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

  m.def("query_closest_points", &QueryClosestPoints);

  m.def("surfel_cave_free_space", &CarveSpace);
  m.def("surfel_merge_redundant", &MergeRedundant);
  m.def("surfel_find_live_to_model_merges", &FindLiveToModelMerges);
  m.def("surfel_find_feat_live_to_model_merges", &FindFeatLiveToModelMerges);

  m.def("raster_indexmap", &RasterIndexmap);
}
