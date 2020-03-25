include_directories(${PYTHON_INCLUDE_DIR})

cuda_add_library(slamtb
  cuda_utils.cpp
  normals.cu
  bilateral_depth_filter.cu
  downsample.cu
  erode_mask.cu
  processing.cpp
  
  camera.cu
  camera.cpp
  
  icp_jacobian_geometric.cu
  icp_jacobian_feature.cu
  icp_jacobian.cpp
  
  exp_rt_to_matrix.cu
  matrix_to_exp_rt.cu
  se3.cpp
  
  aabb.cpp
  sat.cpp
  geometry.cpp
  triangle_mesh_octree.cpp
  nearest_neighbors.cu
  nearest_neighbors.cpp
  fpcl_matcher.cu
  matching.cpp
  
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
  surfel.cpp
  
  elastic_fusion_update.cu
  elastic_fusion_clean.cu
  elastic_fusion.cpp
  
  slamfeat.cpp
  )

target_include_directories(slamtb
  PUBLIC
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/slamtb>
  )

target_link_libraries(slamtb "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)

set_target_properties(slamtb
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET slamtb PROPERTY CXX_STANDARD 14)

add_library(_cslamtb SHARED _cslamtb.cpp)
set_property(TARGET _cslamtb PROPERTY CXX_STANDARD 14)
set_target_properties(_cslamtb PROPERTIES PREFIX "")
set_target_properties(_cslamtb PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_SOURCE_DIR}/slamtb)
target_compile_options(_cslamtb PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(_cslamtb slamtb)

add_subdirectory(test)
