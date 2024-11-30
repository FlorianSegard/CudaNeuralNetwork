#include "Tensor.hpp"

// ----------------------------------------------------------- TRANSPOSE ----------------------------------------------------------- \\

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

// template definitions
template Tensor<float> transposeGPU(const Tensor<float>& input);
template Tensor<double> transposeGPU(const Tensor<double>& input);
template Tensor<int> transposeGPU(const Tensor<int>& input);

// ----------------------------------------------------------- FILL UP WITH ZEROS ----------------------------------------------------------- \\

// Very simple filling up with zeros kernel might not be optimized
template <class T>
__global__ void fillZeroKernel(T* input, int width, int height, size_t inStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        input[x + inStride / sizeof(T) * y] = 0;
    }
}

template <class T>
void fillZeroGPU(Tensor<T>& input) {

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    fillZeroKernel<T><<<gridSize, blockSize>>>(
        input.buffer, input.width, input.height, input.stride
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

}

// template definitions
template void fillZeroGPU(Tensor<float>& input);
template void fillZeroGPU(Tensor<double>& input);
template void fillZeroGPU(Tensor<int>& input);


// ----------------------------------------------------------- DOT ----------------------------------------------------------- \\


// Very simple filling up with zeros kernel might not be optimized
// template <class T>
// __global__ void dotGPUKernel(T* input, T* other, T* result, int width_input, int height_input, int width_output, size_t inputStride, size_t otherStride, size_t resultStride) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (y < height_input && x < width_output) {
//         T sum = 0;
//         for (int k = 0; k < width_input; k++)
//         {
//             T a_val = input[k + y * inputStride / sizeof(T)];
//             T b_val = other[x + k * otherStride / sizeof(T)];
//             sum += a_val * b_val;
//         }
//         result[x + y * otherStride / sizeof(T)] = sum;
//     }
// }


#define TILE_SIZE 16

template <class T>
__global__ void dotGPUKernel(T* input, T* other, T* result,
                                      int width_input, int height_input, int width_output,
                                      size_t inputStride, size_t otherStride, size_t resultStride) {
    __shared__ T shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    T sum = 0;

    size_t inputStrideElements = inputStride / sizeof(T);
    size_t otherStrideElements = otherStride / sizeof(T);
    size_t resultStrideElements = resultStride / sizeof(T);

    for (int t = 0; t < (width_input + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from input matrix into shared memory
        if (row < height_input && t * TILE_SIZE + tx < width_input) {
            shared_A[ty][tx] = input[row * inputStrideElements + t * TILE_SIZE + tx];
        } else {
            shared_A[ty][tx] = 0;
        }

        // Load tile from other matrix into shared memory
        if (col < width_output && t * TILE_SIZE + ty < width_input) {
            shared_B[ty][tx] = other[(t * TILE_SIZE + ty) * otherStrideElements  + col];
        } else {
            shared_B[ty][tx] = 0;
        }

        __syncthreads();

        // Compute partial product for the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < height_input && col < width_output) {
        result[row * resultStrideElements + col] = sum;
    }
}


template <class T>
Tensor<T> dotGPU(const Tensor<T>& input, const Tensor<T>& other) {
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    Tensor<T> result(other.width, input.height, true);

    dotGPUKernel<T><<<gridSize, blockSize>>>(
        input.buffer, other.buffer, result.buffer, 
        input.width, input.height, other.width, 
        input.stride, other.stride, result.stride
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return result;
}

// template definitions
template Tensor<float> dotGPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> dotGPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> dotGPU(const Tensor<int>& input, const Tensor<int>& other);