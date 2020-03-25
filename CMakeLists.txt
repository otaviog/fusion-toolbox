cmake_minimum_required(VERSION 3.8)
project(SlamToolbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(PythonInterp 3 REQUIRED)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/get_torch_dir.py
  OUTPUT_VARIABLE FOUND_TORCH_DIR)
string(STRIP ${FOUND_TORCH_DIR} FOUND_TORCH_DIR)
set(Torch_DIR ${FOUND_TORCH_DIR})

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(OpenMP)

if (NOT WIN32)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  set(CMAKE_CXX_FLAGS -fPIC)
endif (NOT WIN32)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_subdirectory(src)
