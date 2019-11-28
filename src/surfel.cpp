#include "surfel.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

namespace fiontb {
void SurfelOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<SurfelOp>(m, "SurfelOp")
      .def_static("compute_confidences", &SurfelOp::ComputeConfidences)
      .def_static("compute_radii", &SurfelOp::ComputeRadii)
      .def_static("downsample", &SurfelOp::Downsample);
}

void SurfelCloud::Allocate(int64_t size, int64_t feature_size,
                           torch::Device device) {
  torch::TensorOptions opt(device);
  positions = torch::empty({size, 3}, opt.dtype(torch::kFloat32));
  confidences = torch::empty({size}, opt.dtype(torch::kFloat32));
  normals = torch::empty({size, 3}, opt.dtype(torch::kFloat32));
  radii = torch::empty({size}, opt.dtype(torch::kFloat32));
  colors = torch::empty({size, 3}, opt.dtype(torch::kUInt8));
  times = torch::empty({size}, opt.dtype(torch::kInt32));
  if (feature_size > 0)
    features = torch::empty({feature_size, size}, opt.dtype(torch::kFloat32));
  else
    features = torch::empty({0, 0}, opt.dtype(torch::kFloat32));
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

}  // namespace fiontb
