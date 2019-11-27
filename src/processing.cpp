#include "processing.hpp"

#include <torch/csrc/utils/pybind.h>

namespace fiontb {

void Processing::RegisterPybind(pybind11::module &m) {
  pybind11::class_<Processing>(m, "Processing")
      .def_static("bilateral_depth_filter", &BilateralDepthFilter)
      .def_static("estimate_normals", &EstimateNormals)
      .def_static("downsample_xyz", &DownsampleXYZ)
      .def_static("downsample_mask", &DownsampleMask)
      .def_static("erode_mask", &ErodeMask);

  py::enum_<EstimateNormalsMethod>(m, "EstimateNormalsMethod")
      .value("CentralDifferences", EstimateNormalsMethod::kCentralDifferences)
      .value("Average8", EstimateNormalsMethod::kAverage8);

  py::enum_<DownsampleXYZMethod>(m, "DownsampleXYZMethod")
      .value("Nearest", DownsampleXYZMethod::kNearest);
}
}  // namespace fiontb
