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
#include "surfel.hpp"
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

  py::class_<Matching>(m, "Matching")
      .def_static("match_dense_points", &Matching::MatchDensePoints);

  m.def("query_closest_points", &QueryClosestPoints);

  SurfelAllocator::RegisterPybind(m);
  IndexMap::RegisterPybind(m);
  MappedSurfelModel::RegisterPybind(m);
  SurfelCloud::RegisterPybind(m);
  SurfelFusionOp::RegisterPybind(m);
  
  m.def("raster_indexmap", &RasterIndexmap);

  ICPJacobian::RegisterPybind(m);

  ProjectOp::RegisterPybind(m);

  m.def("so3t_exp_op_forward", &SO3tExpOp::Forward);
  m.def("so3t_exp_op_backward", &SO3tExpOp::Backward);

  m.def("featuremap_op_forward", &FeatureMapOp::Forward);
  m.def("featuremap_op_backward", &FeatureMapOp::Backward);

  py::class_<FeatureMap3DOp>(m, "FeatureMap3DOp")
      .def_static("forward", &FeatureMap3DOp::Forward)
      .def_static("compute_epsilon_distances",
                  &FeatureMap3DOp::ComputeEpsilonDistances)
      .def_static("backward", &FeatureMap3DOp::Backward);

  py::enum_<DownsampleXYZMethod>(m, "DownsampleXYZMethod")
      .value("Nearest", DownsampleXYZMethod::kNearest);
  py::class_<Downsample>(m, "Downsample")
      .def_static("downsample_xyz", &Downsample::DownsampleXYZ)
      .def_static("downsample_mask", &Downsample::DownsampleMask);
}
