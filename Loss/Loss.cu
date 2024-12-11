#include "Loss.hpp"

__global__ void sumOfSquaresKernel(const float* input, int width, int height, size_t stride, float* partialSums) {
    int pitch = stride / sizeof(float);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    // Local sum for this thread
    float localSum = (float)0;

    // Stride through the entire tensor, covering different elements
    // Each thread handles multiple elements.
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int y = i / width;
        int x = i % width;

        int input_index = y * pitch + x;
        float val = input[input_index];
        localSum += val * val;
    }

    // Use shared memory for intra-block reduction
    extern __shared__ float shared[];
    int tx = threadIdx.x;
    shared[tx] = localSum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            shared[tx] += shared[tx + s];
        }
        __syncthreads();
    }

    // Add the block's result to the global partial sum
    if (tx == 0) {
        atomicAdd(partialSums, shared[0]);
    }
}

// Templated host function to invoke the kernel and compute sum of squares on the GPU
float sumOfSquaresGPU(const Tensor<float>& input) {
    // Allocate device memory for the partial sum result
    float* d_partialSum;
    cudaMalloc(&d_partialSum, sizeof(float));
    cudaMemset(d_partialSum, 0, sizeof(float));

    // Launch configuration
    int blockSize = 256;
    int gridSize = (input.width * input.height + blockSize - 1) / blockSize;

    // Dynamic shared memory: we pass blockSize * sizeof(T)
    sumOfSquaresKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(input.buffer, input.width, input.height, input.stride, d_partialSum);
    cudaDeviceSynchronize();

    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_partialSum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partialSum);

    return result;
}

