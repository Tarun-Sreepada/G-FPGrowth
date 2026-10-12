cmake_minimum_required(VERSION 3.24)
project(G-FPGrowth LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/spdlog)
add_subdirectory(extern/CLI11)

add_executable(g-fpgrowth
  src/main.cpp
  src/log.cpp
  src/cpu_fpgrowth.cpp
  src/gfpgrowth.cu
  src/parser.cu
  src/fp_tree.cu
  src/miner.cu
  src/eclat_gpu.cu)
target_include_directories(g-fpgrowth PRIVATE include)
set_source_files_properties(src/cpu_fpgrowth.cpp PROPERTIES COMPILE_OPTIONS "-O3;-march=native;-funroll-loops")
find_package(Threads REQUIRED)
set_target_properties(g-fpgrowth PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(g-fpgrowth PRIVATE spdlog::spdlog CLI11::CLI11 CUDA::cudart Threads::Threads)
