// dropout.cu

#include "Dropout.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ctime>


curandState* d_states = nullptr;




__global__ void initCurandStates(curandState* states, unsigned long seed, int width, int height, size_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * (stride / sizeof(float)) + x;
        curand_init(seed, index, 0, &states[index]);
    }
}

void initializeCurandStates(int width, int height, size_t stride) {
    if (!d_states) {
        cudaMalloc(&d_states, width * height * sizeof(curandState));

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        unsigned long seed = static_cast<unsigned long>(time(NULL));
        initCurandStates<<<gridSize, blockSize>>>(d_states, seed, width, height, stride);
        cudaDeviceSynchronize();
    }
}

void freeCurandStates() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}

__global__ void fillMaskKernel(float* mask, curandState* states, float drop_rate, int width, int height, size_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * (stride / sizeof(float)) + x;
        float rand_val = curand_uniform(&states[index]);
        mask[index] = (rand_val > drop_rate) ? 1.0f : 0.0f;
    }
}

void fillMaskGPU(Tensor<float>* mask, float drop_rate) {
    int width = mask->width;
    int height = mask->height;
    size_t stride = mask->stride;


    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Initialize curand states
    initializeCurandStates(width, height, stride);

    // Fill the mask with random values
    fillMaskKernel<<<gridSize, blockSize>>>(mask->buffer, d_states, drop_rate, width, height, stride);
    cudaDeviceSynchronize();

}



