cmake_minimum_required(VERSION 3.18)
project(catdog_inference CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths
set(TENSORRT_ROOT "C:/TensorRT/TensorRT-10.15.1.29")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")

include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
)

add_executable(catdog_inference main.cpp)

target_link_libraries(catdog_inference
    nvinfer_10
    cudart
)