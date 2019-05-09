include_directories(${PYTHON_INCLUDE_DIR})

cuda_add_library(_fiontb SHARED normals.cpp surfel_fusion.cpp _fiontb.cpp
  aabb.cpp aabb.cu
  octtree.cpp)

target_include_directories(_fiontb
  PUBLIC
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/fiontb>
  )

target_link_libraries(_fiontb "${TORCH_LIBRARIES}")
set_target_properties(_fiontb
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET _fiontb PROPERTY CXX_STANDARD 14)
set_target_properties(_fiontb PROPERTIES PREFIX "")
set_target_properties(_fiontb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fiontb)

