cmake_minimum_required(VERSION 3.18)
project(TensorTest CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(first_test_model
    Tensor/Tensor.cpp
    Tensor/Tensor.cu
    Tensor/Tensor.hpp
    Model/Model.hpp
    Optimizer/Optimizer.hpp
    Layers/Layers.hpp
    Layers/Linear/Linear.hpp
    Loss/Loss.cpp
    Loss/Loss.cu
    Loss/Loss.hpp
    main.cpp
)

target_include_directories(first_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(first_test_model CUDA::cudart)