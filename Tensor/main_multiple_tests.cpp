#include <iostream>
#include <cassert>
#include "Tensor.hpp"

// Function to fill a Tensor with sequential values
template <typename T>
void fillTensor(Tensor<T>& tensor) {
    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            tensor[y][x] = static_cast<T>(y * tensor.width + x);
        }
    }
}

// Function to print a Tensor
template <typename T>
void printTensor(const Tensor<T>& tensor) {
    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            std::cout << tensor[y][x] << " ";
        }
        std::cout << "\n";
    }
}

// Test cases
void testTensorCreation() {
    Tensor<int> tensor(3, 4, false);
    assert(tensor.width == 3);
    assert(tensor.height == 4);
    assert(!tensor.device);
    fillTensor(tensor);
    std::cout << "Tensor Creation Test Passed!\n";
}

void testTensorClone() {
    Tensor<int> tensor(3, 4, false);
    fillTensor(tensor);

    Tensor<int> clone = tensor.clone();
    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            assert(clone[y][x] == tensor[y][x]);
        }
    }
    std::cout << "Tensor Clone Test Passed!\n";
}

void testTensorSwitchDevice() {
    Tensor<int> tensor(3, 4, false);
    fillTensor(tensor);

    Tensor<int> gpuTensor = tensor.switchDevice(true);
    Tensor<int> cpuTensor = gpuTensor.switchDevice(false);

    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            assert(cpuTensor[y][x] == tensor[y][x]);
        }
    }
    std::cout << "Tensor Switch Device Test Passed!\n";
}

void testTensorTransposeCPU() {
    Tensor<int> tensor(3, 2, false);
    fillTensor(tensor);

    Tensor<int> transposed = tensor.transpose();

    for (int y = 0; y < transposed.height; ++y) {
        for (int x = 0; x < transposed.width; ++x) {
            assert(transposed[y][x] == tensor[x][y]);
        }
    }
    std::cout << "Tensor Transpose CPU Test Passed!\n";
}

#ifdef __CUDACC__
void testTensorTransposeGPU() {
    Tensor<int> tensor(3, 2, true);
    fillTensor(tensor);

    Tensor<int> transposed = tensor.transpose();

    Tensor<int> cpuTensor = transposed.switchDevice(false);
    Tensor<int> originalCpuTensor = tensor.switchDevice(false);

    for (int y = 0; y < cpuTensor.height; ++y) {
        for (int x = 0; x < cpuTensor.width; ++x) {
            assert(cpuTensor[y][x] == originalCpuTensor[x][y]);
        }
    }
    std::cout << "Tensor Transpose GPU Test Passed!\n";
}
#endif

int main() {
    testTensorCreation();
    testTensorClone();
    testTensorSwitchDevice();
    testTensorTransposeCPU();

#ifdef __CUDACC__
    testTensorTransposeGPU();
#endif

    std::cout << "All tests passed!\n";
    return 0;
}

