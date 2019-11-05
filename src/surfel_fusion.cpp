#include "surfel_fusion.hpp"

#include <cuda_runtime.h>
#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "surfel_fusion_common.hpp"
#include "cuda_utils.hpp"

namespace py = pybind11;

namespace fiontb {

std::mutex MergeMap<kCPU>::mutex_;

void IndexMap::Synchronize() { CudaSafeCall(cudaDeviceSynchronize()); }

void IndexMap::RegisterPybind(py::module &m) {
  py::class_<IndexMap>(m, "IndexMap")
      .def(py::init())
      .def("to", &IndexMap::To)
      .def("synchronize", &IndexMap::Synchronize)
      .def_readwrite("position_confidence", &IndexMap::position_confidence)
      .def_readwrite("normal_radius", &IndexMap::normal_radius)
      .def_readwrite("color", &IndexMap::color)
      .def_readwrite("indexmap", &IndexMap::indexmap)
      .def_property("width", &IndexMap::get_width, nullptr)
      .def_property("height", &IndexMap::get_height, nullptr)
      .def_property("device", &IndexMap::get_device, nullptr);
}

void MappedSurfelModel::RegisterPybind(py::module &m) {
  py::class_<MappedSurfelModel>(m, "MappedSurfelModel")
      .def(py::init())
      .def_readwrite("positions", &MappedSurfelModel::positions)
      .def_readwrite("confidences", &MappedSurfelModel::confidences)
      .def_readwrite("normals", &MappedSurfelModel::normals)
      .def_readwrite("radii", &MappedSurfelModel::radii)
      .def_readwrite("colors", &MappedSurfelModel::colors)
      .def_readwrite("times", &MappedSurfelModel::times)
      .def_readwrite("features", &MappedSurfelModel::features);
}

void SurfelCloud::RegisterPybind(py::module &m) {
  py::class_<SurfelCloud>(m, "SurfelCloud")
      .def(py::init())
      .def_readwrite("positions", &SurfelCloud::positions)
      .def_readwrite("confidences", &SurfelCloud::confidences)
      .def_readwrite("normals", &SurfelCloud::normals)
      .def_readwrite("radii", &SurfelCloud::radii)
      .def_readwrite("colors", &SurfelCloud::colors)
      .def_readwrite("times", &SurfelCloud::times)
      .def_readwrite("features", &SurfelCloud::features)
      .def_property("size", &SurfelCloud::get_size, nullptr);
}

void SurfelFusionOp::RegisterPybind(py::module &m) {
  py::class_<SurfelFusionOp>(m, "SurfelFusionOp")
      .def_static("update", &SurfelFusionOp::Update)
      .def_static("carve_space", &SurfelFusionOp::CarveSpace)
      .def_static("merge", &SurfelFusionOp::Merge)
      .def_static("clean", &SurfelFusionOp::Clean)
      .def_static("copy_features", &SurfelFusionOp::CopyFeatures);
}

}  // namespace fiontb
