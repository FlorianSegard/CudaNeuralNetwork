cmake_minimum_required(VERSION 3.18)

project(TensorTest CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(
    install/include
    /usr/local/onnx/install/include # Add ONNX include directory explicitly
    ${Protobuf_INCLUDE_DIRS}
)

# Library directories
link_directories(
    install/lib
    /usr/local/onnx/install/lib # Add ONNX library directory explicitly
)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Protobuf REQUIRED)

# Debug output for Protobuf
message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")

# Ensure ONNX dependencies are correctly found
find_package(ONNX REQUIRED)
if (ONNX_FOUND)
    message(STATUS "ONNX_FOUND: ${ONNX_FOUND}")
else ()
    message(FATAL_ERROR "ONNX package not found.")
endif()

# Compiler info
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

# Compiler definitions for ONNX
add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=onnx)

# Add executable
cmake_minimum_required(VERSION 3.18)

project(TensorTest CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(
    install/include
    /usr/local/onnx/install/include # Add ONNX include directory explicitly
    ${Protobuf_INCLUDE_DIRS}
)

# Library directories
link_directories(
    install/lib
    /usr/local/onnx/install/lib # Add ONNX library directory explicitly
)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Protobuf REQUIRED)

# Debug output for Protobuf
message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")

# Ensure ONNX dependencies are correctly found
find_package(ONNX REQUIRED)
if (ONNX_FOUND)
    message(STATUS "ONNX_FOUND: ${ONNX_FOUND}")
else ()
    message(FATAL_ERROR "ONNX package not found.")
endif()

# Compiler info
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

# Compiler definitions for ONNX
add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=onnx)

# Add executable
add_executable(first_test_model
    Tensor/Tensor.cpp
    Tensor/Tensor.cu
    Tensor/Tensor.hpp
    Model/Model.hpp
    Optimizer/Optimizer.hpp
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
    Layers/Dropout/Dropout.cpp
    Layers/Dropout/Dropout.cu
    Layers/Dropout/Dropout.hpp
    Logger/Logger.cpp
    Logger/Logger.hpp
    Scheduler/Scheduler.hpp
    mainforbenchmark.cpp
)

# Include the current directory for private includes
target_include_directories(first_test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link ONNX, Protobuf, and CUDA libraries
target_link_libraries(first_test_model
    PRIVATE
    /usr/local/onnx/install/lib/libonnx_proto.a # Explicitly link the static ONNX library
    ${Protobuf_LIBRARIES}
    CUDA::cudart
    CUDA::curand
)
