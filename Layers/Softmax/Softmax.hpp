#pragma once
#include "../Layers.hpp"
#include "../../Tensor/Tensor.hpp"
#include "Logger/Logger.hpp"

Tensor<float> softmaxGPU(Tensor<float>& input);
__global__ void softmaxForwardKernel(float* input, float* output, int width, int height, size_t inStride, size_t outStride);

Tensor<float> softmaxBackwardGPU(Tensor<float>& output, Tensor<float>& dOutput);
__global__ void softmaxBackwardKernel(float* output, float* dOutput, float* dInput, int width, int height, size_t outStride, size_t dOutStride);

Tensor<float> softmaxCPU(Tensor<float>& input);
Tensor<float> softmaxBackwardCPU(Tensor<float>& output, Tensor<float>& dOutput);

struct Softmax : public Layer {
    bool used_with_cat_cross_entropy = false;

    Softmax(bool used_with_cross_entropy, int inputSize = 0, int outputSize = 0, bool device = false, bool require_grad = true)
        : Layer(inputSize, outputSize, device, require_grad) {
        this->used_with_cat_cross_entropy = used_with_cross_entropy;
    }

    // Forward pass for Softmax
    Tensor<float> computeForward(Tensor<float>& input, bool eval) override {
        Logger::infer(">>> Softmax");
        if (input.device == true) {
            return softmaxGPU(input);
        } else {
            return softmaxCPU(input);
        }
    }

    // Backward pass for Softmax
    Tensor<float> backward(Tensor<float>& dOutput) override {
        if (!require_grad) return Tensor<float>();
        Logger::backprop(">>> Softmax");

        if (used_with_cat_cross_entropy) {
            return dOutput.clone();
        }

        if (dOutput.device == true) {
            Tensor<float> t_float = softmaxBackwardGPU(this->lastInput, dOutput);
            Logger::debugTensor(LogLevel::DEBUG, t_float);
            return t_float;
        } else {
            Tensor<float> t_float = softmaxBackwardCPU(this->lastInput, dOutput);
            Logger::debugTensor(LogLevel::DEBUG, t_float);
            return t_float;
        }
    }
};
