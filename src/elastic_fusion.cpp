#include "elastic_fusion.hpp"

#include <torch/csrc/utils/pybind.h>

namespace slamtb {
void ElasticFusionOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<ElasticFusionOp>(m, "ElasticFusionOp")
      .def_static("update", &ElasticFusionOp::Update)
      .def_static("clean", &ElasticFusionOp::Clean);
}
}  // namespace slamtb
