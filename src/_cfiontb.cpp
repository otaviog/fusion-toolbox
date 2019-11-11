#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include "camera.hpp"
#include "elastic_fusion.hpp"
#include "fsf.hpp"
#include "icpodometry.hpp"
#include "matching.hpp"
#include "mesh.hpp"
#include "nearest_neighbors.hpp"
#include "processing.hpp"
#include "so3.hpp"
#include "surfel.hpp"
#include "surfel_fusion.hpp"
#include "trigoctree.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_cfiontb, m) {
  using namespace fiontb;

  // fiontb.frame
  Processing::RegisterPybind(m);

  // fiontb.pose
  ICPJacobian::RegisterPybind(m);
  ProjectOp::RegisterPybind(m);
  RigidTransformOp::RegisterPybind(m);
  SO3tExpOp::RegisterPybind(m);

  // fiontb.spatial
  TrigOctree::RegisterPybind(m);
  FPCLMatcherOp::RegisterPybind(m);
  NearestNeighborsOp::RegisterPybind(m);

  // fiontb.fusion.surfel
  SurfelOp::RegisterPybind(m);
  SurfelAllocator::RegisterPybind(m);
  IndexMap::RegisterPybind(m);
  MappedSurfelModel::RegisterPybind(m);
  SurfelCloud::RegisterPybind(m);
  ElasticFusionOp::RegisterPybind(m);
  SurfelFusionOp::RegisterPybind(m);
  FSFOp::RegisterPybind(m);
}
