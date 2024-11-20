#include <iostream>
#include <vector>
#include <cuda_runtime.h>

extern "C" void launchVectorAdd(const float *A, const float *B, float *C, int numElements);

int main() {
    const int numElements = 50000;

    // Allocate host vectors
    std::vector<float> h_A(numElements, 1.0f);
    std::vector<float> h_B(numElements, 2.0f);
    std::vector<float> h_C(numElements);

    // Allocate device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_B, numElements * sizeof(float));
    cudaMalloc(&d_C, numElements * sizeof(float));

    // Copy input vectors from host to device memory
    cudaMemcpy(d_A, h_A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    launchVectorAdd(d_A, d_B, d_C, numElements);

    // Copy result back to host memory
    cudaMemcpy(h_C.data(), d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (std::abs(h_C[i] - 3.0f) > 1e-5) {
            std::cout << "Verification failed at index " << i << std::endl;
            success = false;
            break;
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print result
    if (success) {
        std::cout << "Vector addition test PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Vector addition test FAILED!" << std::endl;
        return 1;
    }
}