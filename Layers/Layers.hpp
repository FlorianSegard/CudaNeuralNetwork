#pragma once
#include "Tensor.hpp"
#include <iostream>
#include <typeinfo>

struct LayerParams {
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> dWeights;   // Gradients for backward
    Tensor<float> dBiases;    // Gradients for backward
    int inputSize = 0;
    int outputSize = 0;

    LayerParams() = default; // Default constructor
    LayerParams(int inputSize, int outputSize, bool device)
        : weights(inputSize, outputSize, device),
          biases(1, outputSize, device),
          dWeights(inputSize, outputSize, device),
          dBiases(1, outputSize, device),
          inputSize(inputSize), outputSize(outputSize) {}


    // TODO in Tensor.hpp
    void switchDevice(bool device) {
        // Logic to reallocate or transfer data between CPU and GPU
        weights.switchDevice(device);
        biases.switchDevice(device);
        dWeights.switchDevice(device);
        dBiases.switchDevice(device);
    }

};

struct Layer {
    LayerParams* params = nullptr;
    bool device = false;
    bool require_grad = true;

    Layer(LayerParams* params = nullptr, bool device = false)
        : params(std::move(params)), device(device), require_grad(require_grad) {}
    
    virtual ~Layer() = default;

    virtual void forward(float* input, float* output) = 0;

    virtual void backward(float* dOutput, float* dInput) = 0;

    void setDevice(bool device) {
        this->device = device;
        if (params) {
            params->switchDevice(device);
        }
        onDeviceChanged(device);
    }

protected:
    virtual void onDeviceChanged(bool device) {
        const char* layerType = typeid(*this).name(); // Get derived class name
        if (device) {
            std::cout << "Switching " << layerType << " layer to GPU..." << std::endl;
        } else {
            std::cout << "Switching " << layerType << " layer to CPU..." << std::endl;
        }
    }
};

// Example for activation Layer ReLu:
struct ReLu : public Layer {
    ReLu(bool device) : Layer(device) {}

    void forward(float* input, float* output) override { // probably should put Tensor to get input size to check if the size is correct
        for (int i = 0; i < inputSize; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }

    void backward(float* dOutput, float* dInput) override { // probably should put Tensor
        for (int i = 0; i < inputSize; ++i) {
            dInput[i] = dOutput[i] > 0 ? dOutput[i] : 0.0f;
        }
    }
};


