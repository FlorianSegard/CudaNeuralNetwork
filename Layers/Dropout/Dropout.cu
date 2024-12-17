// dropout.cu

#include "Dropout.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ctime>


__global__ void fillMaskKernel(float* mask, curandState* states, float drop_rate,
                              int width, int height, size_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * (stride / sizeof(float)) + x;
        float rand_val = curand_uniform(&states[index]);
        mask[index] = (rand_val >= drop_rate) ? 1.0f : 0.0f;
    }
}

void fillMaskGPU(Tensor<float>* mask, float drop_rate, curandState* states) {
    dim3 blockSize(32, 32);
    dim3 gridSize((mask->width + blockSize.x - 1) / blockSize.x,
                  (mask->height + blockSize.y - 1) / blockSize.y);

    fillMaskKernel<<<gridSize, blockSize>>>(mask->buffer, states, drop_rate,
                                           mask->width, mask->height, mask->stride);
    cudaDeviceSynchronize();
}

__global__ void initCurandStates(curandState* states, unsigned long seed, int width, int height, size_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * (stride / sizeof(float)) + x;
        curand_init(seed + index, 0, 0, &states[index]);
    }
}

void initializeCurandStates(curandState** d_states, int width, int height, size_t stride) {
    size_t total_elements = (stride / sizeof(float)) * height;
    cudaMalloc(d_states, total_elements * sizeof(curandState));

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    auto seed = static_cast<unsigned long>(time(nullptr));
    initCurandStates<<<gridSize, blockSize>>>(*d_states, seed, width, height, stride);
    cudaDeviceSynchronize();
}