#pragma once
#include "Layers.hpp"
#include <iostream>
#include <typeinfo>

// Example for Linear layer:
struct Linear : public Layer {
    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
        : Layer(new LayerParams(inputSize, outputSize, device), device) {
        this->require_grad = require_grad;
    }

    Tensor<float> computeForward(Tensor<float> input) override {
        Tensor<float> output(input.width, params->outputSize, device); // idk about this
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
        return dInput;
    }
};
