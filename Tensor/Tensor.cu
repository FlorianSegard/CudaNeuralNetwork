#include "Tensor.hpp"
#include <curand_kernel.h>

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
        input[y * (inStride / sizeof(T)) + x] = 0;
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

//     if (y < height_input && x < width_output) 
//     {
//         T sum = 0;
//         for (int k = 0; k < width_input; k++) 
//         {
//             T a_val = input[k + y * inputStride / sizeof(T)];
//             T b_val = other[x + k * otherStride / sizeof(T)];
//             sum += a_val * b_val;
//         }
//         result[x + y * resultStride / sizeof(T)] = sum;
//      }
// }



#define TILE_SIZE 32

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
    dim3 blockSize(32, 32);
    dim3 gridSize((other.width + blockSize.x - 1) / blockSize.x,
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


// ----------------------------------------------------------- TERM TO TERM MULT ----------------------------------------------------------- \\

template <class T>
__global__ void termtotermMultKernel(const T* input, const T* other, T* result, int width, int height, size_t inputStride, size_t otherStride, size_t resultStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIndex = x + (inputStride / sizeof(T)) * y;
        int otherIndex = x + (otherStride / sizeof(T)) * y;
        int resultIndex = x + (resultStride / sizeof(T)) * y;

        result[resultIndex] = input[inputIndex] * other[otherIndex];
    }
}

template <class T>
Tensor<T> termtotermMultGPU(const Tensor<T>& input, const Tensor<T>& other) {
    Tensor<T> result(input.width, input.height, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, 
                  (input.height + blockSize.y - 1) / blockSize.y);

    termtotermMultKernel<<<gridSize, blockSize>>>(input.buffer, other.buffer, result.buffer,
                                        result.width, result.height,
                                        input.stride, other.stride, result.stride);

    cudaDeviceSynchronize();
    return result;
}

// template definitions
template Tensor<float> termtotermMultGPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> termtotermMultGPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> termtotermMultGPU(const Tensor<int>& input, const Tensor<int>& other);

// ----------------------------------------------------------- ADD ----------------------------------------------------------- \\

template <class T>
__global__ void addKernel(const T* input, const T* other, T* result, int width, int height, int inputWidth, int inputHeight, int otherWidth, int otherHeight, size_t inputStride, size_t otherStride, size_t resultStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIndex = (x % inputWidth) + (inputStride / sizeof(T)) * (y % inputHeight);
        int otherIndex = (x % otherWidth) + (otherStride / sizeof(T)) * (y % otherHeight);
        int resultIndex = x + (resultStride / sizeof(T)) * y;

        result[resultIndex] = input[inputIndex] + other[otherIndex];
    }
}

template <class T>
Tensor<T> addGPU(const Tensor<T>& input, const Tensor<T>& other) {
    int resultWidth = std::max(input.width, other.width);
    int resultHeight = std::max(input.height, other.height);

    Tensor<T> result(resultWidth, resultHeight, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x, 
                  (input.height + blockSize.y - 1) / blockSize.y);

    addKernel<<<gridSize, blockSize>>>(input.buffer, other.buffer, result.buffer,
                                        resultWidth, resultHeight,
                                        input.width, input.height, other.width, other.height,
                                        input.stride, other.stride, result.stride);

    cudaDeviceSynchronize();
    return result;
}

// template definitions
template Tensor<float> addGPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> addGPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> addGPU(const Tensor<int>& input, const Tensor<int>& other);

// ----------------------------------------------------------- Scalar Mult ----------------------------------------------------------- \\

template <class T>
__global__ void scalarMultiplyKernel(const T* input, T* output, T scalar, int width, int height, size_t inStride, size_t outStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIndex = y * (inStride / sizeof(T)) + x;
        int outputIndex = y * (outStride / sizeof(T)) + x;
        output[outputIndex] = input[inputIndex] * scalar;
    }
}

template <class T>
Tensor<T> scalarMultiplyGPU(const Tensor<T>& input, const T scalar) {
    Tensor<T> result(input.width, input.height, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    scalarMultiplyKernel<<<gridSize, blockSize>>>(input.buffer, result.buffer, scalar,
                                                  input.width, input.height,
                                                  input.stride, result.stride);
    cudaDeviceSynchronize();
    return result;
}

// template definitions
template Tensor<float> scalarMultiplyGPU(const Tensor<float>& input, const float scalar);
template Tensor<double> scalarMultiplyGPU(const Tensor<double>& input, const double scalar);
template Tensor<int> scalarMultiplyGPU(const Tensor<int>& input, const int scalar);

// ----------------------------------------------------------- FILE ONES ----------------------------------------------------------- \\

template <class T>
__global__ void fillOnesKernel(T* input, int width, int height, size_t stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        input[y * (stride / sizeof(T)) + x] = T(1);
    }
}

template <class T>
void fillOnesGPU(Tensor<T>& input) {
    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    fillOnesKernel<T><<<gridSize, blockSize>>>(
            input.buffer, input.width, input.height, input.stride
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

// template definitions
template void fillOnesGPU(Tensor<float>& input);
template void fillOnesGPU(Tensor<double>& input);
template void fillOnesGPU(Tensor<int>& input);

// ----------------------------------------------------------- Clip Gradients ----------------------------------------------------------- \\

template <typename T>
__global__ void clipGradientsKernel(T* gradients, int width, int height, size_t stride, T clipValue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        size_t index = y * (stride / sizeof(T)) + x;
        gradients[index] = max(min(gradients[index], clipValue), -clipValue);
    }
}

template <class T>
void clipGradientsGPU(Tensor<T>& gradients, const T clipValue) {
    dim3 blockSize(32, 32);
    dim3 gridSize(
        (gradients.width + blockSize.x - 1) / blockSize.x,
        (gradients.height + blockSize.y - 1) / blockSize.y
    );

    clipGradientsKernel<<<gridSize, blockSize>>>(
        gradients.buffer,
        gradients.width,
        gradients.height,
        gradients.stride,
        clipValue
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template void clipGradientsGPU(Tensor<float>& gradients, float clipValue);
template void clipGradientsGPU(Tensor<double>& gradients, double clipValue);
template void clipGradientsGPU(Tensor<int>& gradients, int clipValue);


// ----------------------------------------------------------- Xavier Init weight Kernel ----------------------------------------------------------- \\

template <class T>
__global__ void initWeightsKernel(T* weights, int width, int height, size_t stride, float limit, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Initialize CUDA random number generator
        curandState state;
        curand_init(seed + y * width + x, 0, 0, &state);

        // generate random number between -limit and limit
        float random = (2.0f * curand_uniform(&state) - 1.0f) * limit;

        size_t index = y * (stride / sizeof(T)) + x;
        weights[index] = random;
    }
}

template <class T>
void initWeightsGPU(Tensor<T>& weights, float limit) {
    dim3 blockSize(32, 32);
    dim3 gridSize(
        (weights.width + blockSize.x - 1) / blockSize.x,
        (weights.height + blockSize.y - 1) / blockSize.y
    );

    // using time as the seed
    auto seed = static_cast<unsigned int>(time(nullptr));

    initWeightsKernel<<<gridSize, blockSize>>>(
        weights.buffer,
        weights.width,
        weights.height,
        weights.stride,
        limit,
        seed
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template void initWeightsGPU(Tensor<float>& weights, float limit);
template void initWeightsGPU(Tensor<double>& weights, float limit);


// ----------------------------------------------------------- Sum column ----------------------------------------------------------- \\


template <class T>
__global__ void sumColumnsKernel(const T* input, T* output, int width, int height, size_t stride) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    if (col >= width) return; // Ensure we don’t go out of bounds

    T sum = 0;
    for (int row = 0; row < height; ++row) {
        sum += input[row * stride / sizeof(T) + col];
    }
    output[col] = sum;
}


template <class T>
Tensor<T> sumColumnsGPU(Tensor<T>& input) {
    // Create a result tensor for the output (1 row, `width` columns)
    Tensor<T> result(input.width, 1, true);

    // Configure CUDA kernel
    dim3 blockSize(256);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    sumColumnsKernel<T><<<gridSize, blockSize>>>(
        input.buffer, result.buffer, input.width, input.height, input.stride
    );
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return result; // Return the result tensor
}

// Explicit template instantiations
template Tensor<float> sumColumnsGPU(Tensor<float>& input);
template Tensor<double> sumColumnsGPU(Tensor<double>& input);
template Tensor<int> sumColumnsGPU(Tensor<int>& input);



