cmake_minimum_required(VERSION 3.10)
project(llm_cpp_engine)

# 1. 忽略 Python 查找警告
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -fopenmp -ffast-math)

# 2. 【核心修复】：强行从 python 中获取路径
# 我们改用 --cmakedir (你刚才输出的报错提示里显示支持这个)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_DIR_FROM_PYTHON
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 打印一下，方便 debug。你应该能看到一个 site-packages 里的路径
message(STATUS "Pybind11 CMake Dir from Python: ${PYBIND11_DIR_FROM_PYTHON}")

# 3. 【强行指定】：不让 CMake 自己乱找，直接指定配置文件位置
set(pybind11_DIR "${PYBIND11_DIR_FROM_PYTHON}")
find_package(pybind11 REQUIRED)

# 4. 再次确认找到的版本（目标是 2.10+）
message(STATUS "ACTUAL Found pybind11 version: ${pybind11_VERSION}")
message(STATUS "ACTUAL pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")

# 剩下的保持一致
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(OpenMP REQUIRED)

set(SRCS 
    "src/tensor.cpp" 
    "src/ops.cpp" 
    "src/inference.cpp" 
    "src/bindings.cpp"
)

pybind11_add_module(llm_cpp_engine ${SRCS})
target_link_libraries(llm_cpp_engine PRIVATE OpenMP::OpenMP_CXX)
