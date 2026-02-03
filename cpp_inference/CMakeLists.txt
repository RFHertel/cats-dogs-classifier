cmake_minimum_required(VERSION 3.18)
project(catdog_inference CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "C:/TensorRT/TensorRT-10.15.1.29")
set(OPENCV_ROOT "C:/opencv/build")

include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENCV_ROOT}/include
)

link_directories(
    ${TENSORRT_ROOT}/lib
    ${OPENCV_ROOT}/x64/vc16/lib
)

add_executable(catdog_inference main.cpp)

target_link_libraries(catdog_inference
    nvinfer_10
    CUDA::cudart
    opencv_world4120
)