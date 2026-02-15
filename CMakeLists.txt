cmake_minimum_required(VERSION 3.18)
project(llm_cpp_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. 显式关闭 LTO，防止符号(fatbinData)被误删
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# 2. 设置架构
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# ===== pybind11 =====
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
# 防止 pybind11 内部开启 LTO
set(PYBIND11_LTO_CXX_FLAGS "" CACHE INTERNAL "")
FetchContent_MakeAvailable(pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(ENGINE_SRC
    src/attn.cu
    src/embedding.cu
    src/ffn.cu
    src/inference.cu
    src/linear.cu
    src/rmsnorm.cu
    src/rope.cu
    src/tensor.cu
    src/transformer.cu
)

pybind11_add_module(llm_cpp_engine
    src/bindings.cpp
    ${ENGINE_SRC}
)

# 【关键修复】精细控制编译选项
# 1. 对于 C++ 文件 (.cpp)，直接传给 g++
target_compile_options(llm_cpp_engine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -ffast-math>
)

# 2. 对于 CUDA 文件 (.cu)：
#    -O3 传给 nvcc
#    -march=native -fopenmp -ffast-math 通过 -Xcompiler 强制传给宿主 g++
#    这样 g++ 就能识别 AVX 指令了
target_compile_options(llm_cpp_engine PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-fopenmp,-ffast-math>
)

# 关闭 RDC (最简编译模式)
set_target_properties(llm_cpp_engine PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    INTERPROCEDURAL_OPTIMIZATION OFF
)

target_link_libraries(llm_cpp_engine PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
)
