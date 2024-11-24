#include "Tensor.hpp"


// Very simple transpose kernel might not be optimized
template <class T>
__global__ void transposeKernel(const T* input, T* output, int width, int height, size_t inStride, size_t outStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * outStride / sizeof(T) + y] = input[y * inStride / sizeof(T) + x];
    }
}

template <class T>
Tensor<T> transposeGPU(const Tensor<T>& input) {
    Tensor<T> result(input.height, input.width, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    transposeKernel<T><<<gridSize, blockSize>>>(
        input.buffer, result.buffer, input.width, input.height, input.stride, result.stride
    );
    cudaDeviceSynchronize();

    return result;
}

template Tensor<float> transposeGPU(const Tensor<float>& input);
template Tensor<double> transposeGPU(const Tensor<double>& input);
template Tensor<int> transposeGPU(const Tensor<int>& input);