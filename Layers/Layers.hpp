#pragma once
#include "../Tensor/Tensor.hpp"
#include <iostream>
#include <typeinfo>

#include "Logger/Logger.hpp"

struct LayerParams {
    Tensor<float> weights;
    Tensor<float> biases;
    Tensor<float> dWeights;   // Gradients for backward
    Tensor<float> dBiases;    // Gradients for backward
    int inputSize = 0;
    int outputSize = 0;

    LayerParams() = default; // Default constructor
    LayerParams(int inputSize, int outputSize, bool device)
        : weights(outputSize, inputSize, device),
          biases(outputSize, 1, device),
          dWeights(outputSize, inputSize, device),
          dBiases(outputSize, 1, device),
          inputSize(inputSize), outputSize(outputSize) {}


    // TODO in Tensor.hpp
    void switchDevice(bool device) {
        // Logic to reallocate or transfer data between CPU and GPU
        weights = weights.switchDevice(device);
        biases = biases.switchDevice(device);
        dWeights = dWeights.switchDevice(device);
        dBiases = dBiases.switchDevice(device);
    }

    void zeroGrad() {
        // Logic to reallocate or transfer data between CPU and GPU
        dWeights.fillZero();
        dBiases.fillZero();
    }
};

struct Layer {
    LayerParams params;
    bool require_grad = true;
    Tensor<float> lastInput;

    explicit Layer(int inputSize, int outputSize, bool device, bool require_grad = true)
        : params(inputSize, outputSize, device), require_grad(require_grad) {}
    
    virtual ~Layer() = default;

    Tensor<float> forward(Tensor<float>& input, bool eval) {
        lastInput = input.clone();
        return computeForward(input, eval);
    }


    void switchDevice(bool device) {
        params.switchDevice(device);
        onDeviceChanged(device);
    }

    void zeroGrad() {
        params.zeroGrad();
    }

    virtual Tensor<float> backward(Tensor<float>& dOutput) = 0;


protected:
    virtual Tensor<float> computeForward(Tensor<float>& input, bool eval) = 0;

    virtual void onDeviceChanged(bool device) {
        const char* layerType = typeid(*this).name(); // Get derived class name
        if (device) {
            Logger::debug("Switching layer to GPU...");
        } else {
            Logger::debug("Switching layer to CPU...");
        }
    }
};

