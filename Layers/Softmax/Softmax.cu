#include "Softmax.hpp"

// Forward Kernel for Softmax
__global__ void softmaxForwardKernel(float* input, float* output, int width, int height, size_t inStride, size_t outStride) {
    extern __shared__ float sharedData[];

    int row = blockIdx.x; // Each block handles one row
    int tid = threadIdx.x;

    if (row >= height) return;

    float* rowInput = input + row * inStride / sizeof(float);
    float* rowOutput = output + row * outStride / sizeof(float);

    // Step 1: Find the maximum value for numerical stability
    float maxVal = -3.402823466e+38F; // Smallest float value
    for (int i = tid; i < width; i += blockDim.x) {
        maxVal = fmaxf(maxVal, rowInput[i]);
    }

    // Perform reduction to find the maximum value across threads
    sharedData[tid] = maxVal;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] = fmaxf(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }
    maxVal = sharedData[0]; // Max value for the row

    // Step 2: Compute exponentials and their sum
    float sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        rowOutput[i] = expf(rowInput[i] - maxVal); // Subtract maxVal for numerical stability
        sum += rowOutput[i];
    }

    // Perform reduction to calculate the sum of exponentials
    sharedData[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    sum = sharedData[0]; // Total sum of exponentials

    // Step 3: Normalize each value
    for (int i = tid; i < width; i += blockDim.x) {
        rowOutput[i] /= sum;
    }
}

// Backward Kernel for Softmax
__global__ void softmaxBackwardKernel(float* output, float* dOutput, float* dInput, int width, int height, size_t outStride, size_t dOutStride) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= height) return;

    extern __shared__ float sharedData[];
    float* rowOutput  = output  + (row * outStride / sizeof(float));
    float* rowDOutput = dOutput + (row * dOutStride / sizeof(float));
    float* rowDInput  = dInput  + (row * dOutStride / sizeof(float));

    // 1. Compute sumVal = sum over j of (y_j * dY_j)
    float partialSum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        partialSum += rowOutput[i] * rowDOutput[i];
    }

    sharedData[tid] = partialSum;
    __syncthreads();

    // Parallel reduction to get the sumVal
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    float sumVal = sharedData[0];

    // 2. Compute dZ_i = y_i * (dY_i - sumVal)
    for (int i = tid; i < width; i += blockDim.x) {
        float Yi = rowOutput[i];
        float dYi = rowDOutput[i];
        rowDInput[i] = Yi * (dYi - sumVal);
    }
}

// Forward Pass on GPU
Tensor<float> softmaxGPU(Tensor<float>& input) {
    Tensor<float> output(input.width, input.height, true);

    int blockSize = 256;
    int gridSize = input.height;
    size_t sharedMemSize = blockSize * sizeof(float);

    softmaxForwardKernel<<<gridSize, blockSize, sharedMemSize>>>(
        input.buffer, output.buffer, input.width, input.height, input.stride, output.stride);

    cudaDeviceSynchronize();
    return output;
}

// Backward Pass on GPU
Tensor<float> softmaxBackwardGPU(Tensor<float>& output, Tensor<float>& dOutput) {
    Tensor<float> dInput(output.width, output.height, true);

    int blockSize = 256;
    int gridSize = output.height;

    softmaxBackwardKernel<<<gridSize, blockSize>>>(
        output.buffer, dOutput.buffer, dInput.buffer, output.width, output.height, output.stride, dOutput.stride);

    cudaDeviceSynchronize();
    return dInput;
}
