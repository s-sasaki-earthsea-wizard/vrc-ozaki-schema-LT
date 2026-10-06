cmake_minimum_required(VERSION 3.22)
project(cpu_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_executable(cpu_bench cpu_bench.cpp)
target_link_libraries(cpu_bench PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
# -march=native: the binary is built and run on the benchmark host (AVX2 + FMA here).
# No -ffast-math, so floating-point semantics stay IEEE (FMA contraction is allowed).
target_compile_options(cpu_bench PRIVATE -O3 -march=native -DNDEBUG)
