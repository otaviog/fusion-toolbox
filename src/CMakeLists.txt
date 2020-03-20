include_directories(${PYTHON_INCLUDE_DIR})

cuda_add_library(fiontb
  cuda_utils.cpp
  normals.cu
  bilateral_depth_filter.cu
  downsample.cu
  erode_mask.cu
  processing.cpp
  camera.cpp
  camera.cu
  icp_jacobian.cpp
  icp_jacobian_geometric.cu
  icp_jacobian_feature.cu
  exp_rt_to_matrix.cu
  exp_rt_transform.cpp
  quat_rt_transform.cpp
  se3.cpp
  aabb.cpp
  sat.cpp
  geometry.cpp
  triangle_mesh_octree.cpp
  nearest_neighbors.cu
  nearest_neighbors.cpp
  fpcl_matcher.cu
  matching.cpp
  surfel.cpp
  surfel.cu
  # surfel_downsample.cu
  surfel_volume.cpp
  surfel_volume.cu
  surfel_allocator.cpp
  surfel_fusion.cpp
  surfel_fusion_update.cu
  surfel_fusion_carve_space.cu
  surfel_fusion_merge.cu
  surfel_fusion_clean.cu
  surfel_fusion_copy_features.cu
  elastic_fusion.cpp
  elastic_fusion_update.cu
  elastic_fusion_clean.cu
  slamfeat.cpp
  )

target_include_directories(fiontb
  PUBLIC
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/fiontb>
  )

target_link_libraries(fiontb "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX Sophus::Sophus)

set_target_properties(fiontb
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET fiontb PROPERTY CXX_STANDARD 14)

add_library(_cfiontb SHARED _cfiontb.cpp)
set_property(TARGET _cfiontb PROPERTY CXX_STANDARD 14)
set_target_properties(_cfiontb PROPERTIES PREFIX "")
set_target_properties(_cfiontb PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/fiontb)
target_compile_options(_cfiontb PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(_cfiontb fiontb)

add_subdirectory(test)
