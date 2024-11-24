#include "Tensor.hpp"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    try {
        // Create a tensor on the host (CPU)
        Tensor<float> hostTensor(4, 3, false); // 4x3 matrix on CPU

        // Fill the tensor with some values
        for (int y = 0; y < hostTensor.height; ++y) {
            for (int x = 0; x < hostTensor.width; ++x) {
                hostTensor[y][x] = static_cast<float>(y * hostTensor.width + x);
            }
        }

        // Print the host tensor
        std::cout << "Host Tensor (Original):" << std::endl;
        for (int y = 0; y < hostTensor.height; ++y) {
            for (int x = 0; x < hostTensor.width; ++x) {
                std::cout << hostTensor[y][x] << " ";
            }
            std::cout << std::endl;
        }

        // Transfer to GPU
        Tensor<float> gpuTensor = hostTensor.switchDevice(true);

        // Transpose on GPU
        Tensor<float> gpuTransposed = gpuTensor.transpose();

        // Transfer back to CPU
        Tensor<float> hostTransposed = gpuTransposed.switchDevice(false);

        // Print the transposed tensor
        std::cout << "Host Tensor (Transposed):" << std::endl;
        for (int y = 0; y < hostTransposed.height; ++y) {
            for (int x = 0; x < hostTransposed.width; ++x) {
                std::cout << hostTransposed[y][x] << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}