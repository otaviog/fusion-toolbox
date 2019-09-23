#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "camera.hpp"
#include "dense_volume.hpp"
#include "downsample.hpp"
#include "filtering.hpp"
#include "icpodometry.hpp"
#include "indexmap.hpp"
#include "matching.hpp"
#include "metrics.hpp"
#include "normals.hpp"
#include "so3.hpp"
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

  m.def("bilateral_depth_filter", &BilateralDepthFilter);

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

  m.def("match_dense_points", &MatchDensePoints);

  m.def("query_closest_points", &QueryClosestPoints);

  m.def("surfel_cave_free_space", &CarveSpace);
  m.def("surfel_find_mergeable_surfels", &FindMergeableSurfels);

  m.def("surfel_find_live_to_model_merges", &FindLiveToModelMerges);
  m.def("surfel_find_feat_live_to_model_merges", &FindFeatLiveToModelMerges);

  py::class_<IndexMap>(m, "IndexMap")
      .def(py::init())
      .def("to", &IndexMap::To)
      .def_readwrite("position_confidence", &IndexMap::position_confidence)
      .def_readwrite("normal_radius", &IndexMap::normal_radius)
      .def_readwrite("color", &IndexMap::color)
      .def_readwrite("indexmap", &IndexMap::indexmap)
      .def_property("width", &IndexMap::get_width, nullptr)
      .def_property("height", &IndexMap::get_height, nullptr)
      .def_property("device", &IndexMap::get_device, nullptr);
      

  py::class_<MappedSurfelModel>(m, "MappedSurfelModel")
      .def(py::init())
      .def_readwrite("positions", &MappedSurfelModel::positions)
      .def_readwrite("confidences", &MappedSurfelModel::confidences)
      .def_readwrite("normals", &MappedSurfelModel::normals)
      .def_readwrite("radii", &MappedSurfelModel::radii)
      .def_readwrite("colors", &MappedSurfelModel::colors);

  py::class_<FeatSurfel>(m, "FeatSurfel")
      .def_static("merge_live", &FeatSurfel::MergeLive);

  m.def("raster_indexmap", &RasterIndexmap);

  py::class_<ICPJacobian>(m, "ICPJacobian")
      .def_static("estimate_geometric", &ICPJacobian::EstimateGeometric)
      .def_static("estimate_intensity", &ICPJacobian::EstimateIntensity);

  m.def("calc_sobel_gradient", &CalcSobelGradient);

  m.def("project_op_forward", &ProjectOp::Forward);
  m.def("project_op_backward", &ProjectOp::Backward);

  m.def("so3t_exp_op_forward", &SO3tExpOp::Forward);
  m.def("so3t_exp_op_backward", &SO3tExpOp::Backward);

  m.def("featuremap_op_forward", &FeatureMapOp::Forward);
  m.def("featuremap_op_backward", &FeatureMapOp::Backward);

  py::enum_<DownsampleXYZMethod>(m, "DownsampleXYZMethod")
      .value("Nearest", DownsampleXYZMethod::kNearest);
  m.def("downsample_xyz", &DownsampleXYZ);
}
