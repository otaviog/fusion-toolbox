#include <memory>

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "camera.hpp"
#include "elastic_fusion.hpp"
#include "icp_jacobian.hpp"
#include "matching.hpp"
#include "nearest_neighbors.hpp"
#include "processing.hpp"
#include "se3.hpp"
#include "slamfeat.hpp"
#include "surfel.hpp"
#include "surfel_fusion.hpp"
#include "surfel_volume.hpp"
#include "triangle_mesh_octree.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_cfiontb, m) {
  using namespace fiontb;

  // fiontb.frame
  Processing::RegisterPybind(m);

  // Transform
  ProjectOp::RegisterPybind(m);
  RigidTransformOp::RegisterPybind(m);
  ExpRtToMatrixOp::RegisterPybind(m);
  MatrixToExpRtOp::RegisterPybind(m);
  
  // Registraion
  ICPJacobian::RegisterPybind(m);

  // Spatial
  TriangleMeshOctree::RegisterPybind(m);
  FPCLMatcherOp::RegisterPybind(m);
  NearestNeighborsOp::RegisterPybind(m);

  // Surfel
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
