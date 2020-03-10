#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "camera.hpp"
#include "elastic_fusion.hpp"
#include "icp_jacobian.hpp"
#include "matching.hpp"
#include "nearest_neighbors.hpp"
#include "processing.hpp"
#include "so3.hpp"
#include "surfel.hpp"
#include "surfel_fusion.hpp"
#include "surfel_volume.hpp"
#include "triangle_mesh_octree.hpp"
#include "slamfeat.hpp"

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
  TriangleMeshOctree::RegisterPybind(m);
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
  SurfelVolume::RegisterPybind(m);

  // etc
  SlamFeatOp::RegisterPybind(m);
  
}
