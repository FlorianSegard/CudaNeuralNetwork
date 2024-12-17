#pragma once
#include "../Layers.hpp"
#include "Logger/Logger.hpp"

Tensor<float> sigmoidCPU(Tensor<float>& input);
Tensor<float> sigmoidBackwardCPU(Tensor<float>& output, Tensor<float>& dOutput);

Tensor<float> sigmoidGPU(Tensor<float>& input);
__global__ void sigmoidForwardKernel(float* input, float* output, int width, int height,
                                    size_t inStride, size_t outStride);

Tensor<float> sigmoidBackwardGPU(Tensor<float>& output, Tensor<float>& dOutput);
__global__ void sigmoidBackwardKernel(float* output, float* dOutput, float* dInput,
                                     int width, int height,
                                     size_t outStride, size_t dOutStride, size_t dInStride);

struct Sigmoid : public Layer {
    Sigmoid(int inputSize = 0, int outputSize = 0, bool device = false, bool require_grad = true)
        : Layer(inputSize, outputSize, device, require_grad) {}

    Tensor<float> computeForward(Tensor<float>& input) override {
        Logger::infer(">>> Sigmoid");
        if (input.device) {
            return sigmoidGPU(input);
        } else {
            return sigmoidCPU(input);
        }
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        if (!require_grad) return Tensor<float>();
        Logger::backprop(">>> Sigmoid");

        if (dOutput.device) {
            return sigmoidBackwardGPU(this->lastInput, dOutput);
        } else {
            return sigmoidBackwardCPU(this->lastInput, dOutput);
        }
    }
};