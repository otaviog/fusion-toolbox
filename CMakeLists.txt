cmake_minimum_required(VERSION 3.8)
project(fiontb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/3rd-party/libtorch)
  message("Downloading libtorch")
  file(DOWNLOAD https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip
	${CMAKE_SOURCE_DIR}/3rd-party/libtorch-shared-with-deps-latest.zip SHOW_PROGRESS)
  message("Downloaded libtorch")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf libtorch-shared-with-deps-latest.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party)
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(Torch_DIR ${CMAKE_SOURCE_DIR}/3rd-party/libtorch/share/cmake/Torch)

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/3rd-party/Catch2)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf Catch2.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party)
endif()

find_package(Torch REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

find_package(OpenMP)

set(CMAKE_CXX_FLAGS -fPIC)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(src)

