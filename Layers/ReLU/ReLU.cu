#include "ReLU.hpp"

// ----------------------------------------------------------- FORWARD ----------------------------------------------------------- \\


__global__ void reluKernel(float* input, float* output, int width, int height, size_t inStride, size_t outStride) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int input_index = y * (inStride / sizeof(float)) + x;
        int output_index = y * (outStride / sizeof(float)) + x;

        output[output_index] = 0.0f;
        if (input[input_index] > 0.0f) {
            output[output_index] = input[input_index];
        }
    }
}

Tensor<float> reluGPU(Tensor<float>& input) 
{
    Tensor<float> result(input.width, input.height, true); // Create a result tensor on GPU

    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    reluKernel<<<gridSize, blockSize>>>(input.buffer, result.buffer,
                                        input.width, input.height,
                                        input.stride, result.stride);

    cudaDeviceSynchronize();
    return result;
}


// ----------------------------------------------------------- BACKWARD ----------------------------------------------------------- \\


__global__ void reluBackwardKernel(float* input, float* dOutput, float* dInput,
                                   int width, int height, size_t inStride, size_t outStride) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int input_index = y * (inStride / sizeof(float)) + x;
        int output_index = y * (outStride / sizeof(float)) + x;
        dInput[output_index] = 0.0f;
        if (input[input_index] > 0.0f) {
            dInput[output_index] = dOutput[output_index];
        }
    }
}

Tensor<float> reluBackwardGPU(Tensor<float>& input, Tensor<float>& dOutput) 
{
    Tensor<float> dInput(input.width, input.height, true);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    reluBackwardKernel<<<gridSize, blockSize>>>(input.buffer, dOutput.buffer, dInput.buffer,
                                                input.width, input.height,
                                                input.stride, dInput.stride);

    cudaDeviceSynchronize();
    return dInput;
}
