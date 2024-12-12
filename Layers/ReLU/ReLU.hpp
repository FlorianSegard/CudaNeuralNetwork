#pragma once
#include "../Layers.hpp"

Tensor<float> reluGPU(Tensor<float>& input);
__global__ void reluKernel(float* input, float* output, int width, int height, size_t inStride, size_t outStride);

Tensor<float> reluCPU(Tensor<float>& input);

Tensor<float> reluBackwardGPU(Tensor<float>& input, Tensor<float>& dOutput);
__global__ void reluBackwardKernel(float* input, float* dOutput, float* dInput, int width, int height, size_t inStride, size_t outStride);

Tensor<float> reluBackwardCPU(Tensor<float>& input, Tensor<float>& dOutput);

struct ReLU : public Layer {
    ReLU(int inputSize = 0, int outputSize = 0, bool device = false, bool require_grad = true)
        : Layer(inputSize, outputSize, device, require_grad) {}

    // Forward pass for ReLU
    Tensor<float> computeForward(Tensor<float>& input) override {
        if (input.device == true) {
            return reluGPU(input);
        }
        else {
            return reluCPU(input);
        }
    }

    // Backward pass for ReLU
    Tensor<float> backward(Tensor<float>& dOutput) override {
        if (!require_grad) return Tensor<float>();

        if (dOutput.device == true) {
            return reluBackwardGPU(this->lastInput, dOutput);
        } else {
            return reluBackwardCPU(this->lastInput, dOutput);
        }
    }
};
