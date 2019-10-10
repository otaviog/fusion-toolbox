include_directories(${PYTHON_INCLUDE_DIR})

cuda_add_library(fiontb
  feature_map3d.cu
  cuda_utils.cpp
  normals.cu
  bilateralfilter.cu
  feature_map.cu
  aabb.cpp
  sat.cpp metrics.cpp
  trigoctree.cpp 
  dense_volume.cpp
  sparse_volume.cpp
  tsdf_fusion.cpp
  camera.cu
  so3.cu
  surfel.cpp
  surfel_allocator.cpp
  surfel_fusion.cpp
  surfel_fusion_merge_live.cu
  surfel_fusion_carve_space.cu
  surfel_fusion_merge.cu
  surfel_fusion_copy_features.cu
  kernel_indexmap.cu  
  icp_jacobian.cu
  downsample.cu
  dense_matcher.cu
  surfel_allocator.cpp
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
set_target_properties(_cfiontb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fiontb)
target_compile_options(_cfiontb PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(_cfiontb fiontb)

add_subdirectory(test)
