#include "Sigmoid.hpp"

__global__ void sigmoidForwardKernel(float* input, float* output, int width, int height,
                                    size_t inStride, size_t outStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int input_index = y * (inStride / sizeof(float)) + x;
        int output_index = y * (outStride / sizeof(float)) + x;

        float val = input[input_index];
        output[output_index] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void sigmoidBackwardKernel(float* output, float* dOutput, float* dInput,
                                     int width, int height,
                                     size_t outStride, size_t dOutStride, size_t dInStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int out_index = y * (outStride / sizeof(float)) + x;
        int dout_index = y * (dOutStride / sizeof(float)) + x;
        int din_index = y * (dInStride / sizeof(float)) + x;

        float sigmoid_x = output[out_index];
        dInput[din_index] = sigmoid_x * (1.0f - sigmoid_x) * dOutput[dout_index];
    }
}

Tensor<float> sigmoidGPU(Tensor<float>& input) {
    Tensor<float> output(input.width, input.height, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);

    sigmoidForwardKernel<<<gridSize, blockSize>>>(
        input.buffer, output.buffer,
        input.width, input.height,
        input.stride, output.stride
    );

    cudaDeviceSynchronize();
    return output;
}

Tensor<float> sigmoidBackwardGPU(Tensor<float>& output, Tensor<float>& dOutput) {
    Tensor<float> dInput(output.width, output.height, true);

    dim3 blockSize(32, 32);
    dim3 gridSize((output.width + blockSize.x - 1) / blockSize.x,
                  (output.height + blockSize.y - 1) / blockSize.y);

    sigmoidBackwardKernel<<<gridSize, blockSize>>>(
        output.buffer, dOutput.buffer, dInput.buffer,
        output.width, output.height,
        output.stride, dOutput.stride, dInput.stride
    );

    cudaDeviceSynchronize();
    return dInput;
}