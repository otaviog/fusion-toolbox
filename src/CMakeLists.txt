include_directories(${PYTHON_INCLUDE_DIR})

add_library(_fiontb SHARED _fiontb.cpp normals.cpp surfel_fusion.cpp)

target_include_directories(_fiontb
  PUBLIC  
  $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/fiontb>
  )

target_link_libraries(_fiontb "${TORCH_LIBRARIES}")

set_property(TARGET _fiontb PROPERTY CXX_STANDARD 14)
set_target_properties(_fiontb PROPERTIES PREFIX "")
set_target_properties(_fiontb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fiontb)
