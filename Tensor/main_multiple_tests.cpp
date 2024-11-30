#include <iostream>
#include <cassert>
#include <chrono>
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

template <typename T>
bool isZeroFilled(const Tensor<T>& tensor) {
    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            if (tensor[y][x] != static_cast<T>(0)) {
                return false;
            }
        }
    }
    return true;
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

void testTensorTransposeGPU() {
    Tensor<int> tensor(3, 2, false);
    fillTensor(tensor);

    Tensor<int> transposed = tensor.transpose();

    Tensor<int> gpuTensor = transposed.switchDevice(true);

    Tensor<int> gpuTensortransposed = gpuTensor.transpose();
    Tensor<int> originalCpuTensor = gpuTensortransposed.switchDevice(false);

    for (int y = 0; y < tensor.height; ++y) {
        for (int x = 0; x < tensor.width; ++x) {
            assert(tensor[y][x] == originalCpuTensor[y][x]);
        }
    }
    std::cout << "Tensor Transpose GPU Test Passed!\n";
}


void testTensorFillZeroAndSwitchDevice() {
    Tensor<int> tensor(3, 4, false);
    tensor.fillZero();

    assert(isZeroFilled(tensor));

    Tensor<int> gpuTensor = tensor.switchDevice(true);
    Tensor<int> cpuTensor = gpuTensor.switchDevice(false);

    assert(isZeroFilled(cpuTensor));

    Tensor<int> gpuTensorZeros = cpuTensor.switchDevice(true);
    gpuTensorZeros.fillZero();
    Tensor<int> cpuTensorZeros = gpuTensorZeros.switchDevice(false);

    assert(isZeroFilled(cpuTensorZeros));

    std::cout << "Tensor Fill Zero and Switch Device Test Passed!\n";
}


void testTensorDOT() {
    Tensor<int> A(120, 20, false);
    Tensor<int> B(15, 120, false);

    std::cout << "Matrix A: " << A.height << " x " << A.width << std::endl;
    std::cout << "Matrix B: " << B.height << " x " << B.width << std::endl;


    fillTensor(A);
    fillTensor(B);

    // printTensor(A);
    // printTensor(B);

    auto startTime = std::chrono::high_resolution_clock::now();
    Tensor<int> C = A.dot(B);
    auto endTime = std::chrono::high_resolution_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;

    printTensor(C);


    Tensor<int> Agpu = A.switchDevice(true);
    Tensor<int> Bgpu = B.switchDevice(true);

    startTime = std::chrono::high_resolution_clock::now();
    Tensor<int> Cgpu = Agpu.dot(Bgpu);
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time elapsed: " << duration << " ms" << std::endl;


    Tensor<int> Cgpucpu = Cgpu.switchDevice(false);
    std::cout << "Matrix C: " << C.height << " x " << C.width << std::endl;

    printTensor(Cgpucpu);

    std::cout << "Tensor Creation Test Passed!\n";
}


void testTensorDOT2() {
    Tensor<int> A(5000, 20000, false);
    Tensor<int> B(15000, 5000, false);

    std::cout << "Matrix A: " << A.height << " x " << A.width << std::endl;
    std::cout << "Matrix B: " << B.height << " x " << B.width << std::endl;


    A.fillZero();
    B.fillZero();

    // printTensor(A);
    // printTensor(B);

    auto startTime = std::chrono::high_resolution_clock::now();
    // Tensor<int> C = A.dot(B);
    auto endTime = std::chrono::high_resolution_clock::now();


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Time elapsed CPU: " << duration << " ms" << std::endl;

    // printTensor(C);


    Tensor<int> Agpu = A.switchDevice(true);
    Tensor<int> Bgpu = B.switchDevice(true);

    startTime = std::chrono::high_resolution_clock::now();
    Tensor<int> Cgpu = Agpu.dot(Bgpu);
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    std::cout << "Time elapsed GPU naiv: " << duration << " ms" << std::endl;


    Tensor<int> Cgpucpu = Cgpu.switchDevice(false);
    std::cout << "Matrix C: " << Cgpucpu.height << " x " << Cgpucpu.width << std::endl;

    // printTensor(Cgpucpu);

    std::cout << "Tensor Creation Test Passed!\n";
}





int main() {
    testTensorCreation();
    testTensorClone();
    testTensorSwitchDevice();
    testTensorTransposeCPU();
    testTensorTransposeGPU();
    testTensorFillZeroAndSwitchDevice();
    // testTensorDOT();
    testTensorDOT2();
    std::cout << "All tests passed!\n";
    return 0;
}

