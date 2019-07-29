include_directories(${PYTHON_INCLUDE_DIR})

cuda_add_library(fiontb
  cuda_utils.cpp
  normals.cpp normals.cu
  filtering.cpp filtering.cu
  aabb.cpp
  sat.cpp metrics.cpp
  trigoctree.cpp 
  dense_volume.cpp
  sparse_volume.cpp
  tsdf_fusion.cpp
  kernel_surfel_intra_merge.cu
  kernel_surfel_live_merge.cu
  kernel_surfel_space_carving.cu
  kernel_indexmap.cu
  kernel_icp_estimate.cu
  kernel_downsample.cu)

target_include_directories(fiontb
  PUBLIC
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/fiontb>
  )

target_link_libraries(fiontb "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)

set_target_properties(fiontb
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET fiontb PROPERTY CXX_STANDARD 14)

add_library(_cfiontb SHARED _fiontb.cpp)
set_property(TARGET _cfiontb PROPERTY CXX_STANDARD 14)
set_target_properties(_cfiontb PROPERTIES PREFIX "")
set_target_properties(_cfiontb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fiontb)
target_compile_options(_cfiontb PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(_cfiontb fiontb)

add_subdirectory(test)
