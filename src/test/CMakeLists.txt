include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_SOURCE_DIR}/3rd-party/Catch2)

add_executable(test_fiontb
  test_aabb.cpp
  catch_main.cpp)
set_property(TARGET test_fiontb PROPERTY CXX_STANDARD 14)
target_link_libraries(test_fiontb fiontb "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)

