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
    Loader/ONNXLoader.hpp
    Loader/TabularLoader.hpp
    Layers/Layers.hpp
    Layers/Linear/Linear.hpp
    Layers/ReLU/ReLU.cpp
    Layers/ReLU/ReLU.cu
    Layers/ReLU/ReLU.hpp
    Layers/Softmax/Softmax.cpp
    Layers/Softmax/Softmax.cu
    Layers/Softmax/Softmax.hpp
    Layers/Sigmoid/Sigmoid.cpp
    Layers/Sigmoid/Sigmoid.cu
    Layers/Sigmoid/Sigmoid.hpp
    Loss/BinaryCrossEntropy/BinCrossEntropy.cpp
    Loss/BinaryCrossEntropy/BinCrossEntropy.cu
    Loss/BinaryCrossEntropy/BinCrossEntropy.hpp
    Loss/CategoricalCrossEntropy/CatCrossEntropy.cpp
    Loss/CategoricalCrossEntropy/CatCrossEntropy.cu
    Loss/CategoricalCrossEntropy/CatCrossEntropy.hpp
    Loss/MeanSquaredError/Mse.cpp
    Loss/MeanSquaredError/Mse.cu
    Loss/MeanSquaredError/Mse.hpp
    Logger/Logger.cpp
    Logger/Logger.hpp
    main2.cpp
)

target_include_directories(first_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(first_test_model CUDA::cudart onnx_proto ${Protobuf_LIBRARIES})