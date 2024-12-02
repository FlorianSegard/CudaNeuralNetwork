cmake_minimum_required(VERSION 3.18)
project(TensorTest CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(tensor_test
    Tensor/main_multiple_tests.cpp
    Tensor/Tensor.cpp
    Tensor/Tensor.cu
    Tensor/Tensor.hpp
)

target_include_directories(tensor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor_test CUDA::cudart)