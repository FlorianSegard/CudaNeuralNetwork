#pragma once
#include "Layers.hpp"
#include <iostream>
#include <typeinfo>

// Example for Linear layer:
struct Linear : public Layer {
    LayerParams params;

    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
        : Layer(device), params(inputSize, outputSize, device), require_grad(require_grad) {}

    Tensor<float> forward(Tensor<float> input, Tensor<float> output) override { // probably should put Tensor to get input size to check if the size is correct
        if (device) {
            forwardLinearGPU(input, output);
        } else {
            forwardLinearCPU(input, output);
        }
    }

    void backward(Tensor<float> dOutput, Tensor<float> dInput) override { // probably should put Tensor
        if (device) {
            backwardLinearGPU(dOutput, dInput);
        } else {
            backwardLinearCPU(dOutput, dInput);
        }
    }
};
