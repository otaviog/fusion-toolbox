#include "surfel_volume.hpp"

#include <cuda_runtime.h>
#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

namespace fiontb {

void SurfelVolume::RegisterPybind(pybind11::module &m) {
  pybind11::class_<SurfelVolume>(m, "SurfelVolume")
      .def(pybind11::init<const Eigen::Vector3f &, const Eigen::Vector3f &,
           float, int>())
      .def("merge", &SurfelVolume::Merge)
      .def("to_surfel_cloud", &SurfelVolume::ToSurfelCloud);
}

}  // namespace fiontb
