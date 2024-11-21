cmake_minimum_required(VERSION 3.18)
project(VectorAddition CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

add_executable(vector_add_test
        vector_add_kernel.cu
        vector_add_test.cpp
)

target_link_libraries(vector_add_test CUDA::cudart)