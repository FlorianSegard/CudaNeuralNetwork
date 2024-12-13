cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "native")

project(TensorTest CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(
        install/include
        ${Protobuf_INCLUDE_DIRS}
)
link_directories(
        install/lib
)
set(CMAKE_PREFIX_PATH install/lib/cmake)

find_package(CUDAToolkit REQUIRED)
find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=onnx)

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
    Loader/ModelLoader.hpp
    main.cpp
)

target_include_directories(first_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(first_test_model CUDA::cudart onnx_proto ${Protobuf_LIBRARIES})