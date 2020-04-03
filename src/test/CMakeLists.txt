include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_SOURCE_DIR}/3rd-party/Catch2)

add_executable(test_slamtb
  test_aabb.cpp
  catch_main.cpp)
set_property(TARGET test_slamtb PROPERTY CXX_STANDARD 17)
target_link_libraries(test_slamtb slamtb
  "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)

