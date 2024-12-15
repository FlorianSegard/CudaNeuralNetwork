#pragma once

#include <cfloat>

#include "../Layers.hpp"
#include "Logger/Logger.hpp"

struct Linear : public Layer {

    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
            : Layer(inputSize, outputSize, device, require_grad) {
        this->params.weights.initializeWeights(inputSize, outputSize);
        this->params.biases.fillZero();
    }

    Tensor<float> computeForward(Tensor<float>& input) override {
        // Linear forward: output = input @ weights.T + biases
        Logger::infer(">>> Linear");

        // Dot matrix product [batch_size, input_size] @ [input_size, output_size]
        Tensor<float> output = input.dot(params.weights);

        // Add biases
        return output + params.biases;
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        // dInput = dOutput @ weights
        Logger::backprop(">>> Linear");
        if (!require_grad) return Tensor<float>();

        Tensor<float> dInput = dOutput.dot(params.weights.transpose());
        Logger::backprop("| -> dInput");
        Logger::debugTensor(LogLevel::BACKPROP, dInput);

        // dWeights = input.T @ dOutput

        Tensor<float> inputT = this->lastInput.transpose();
        params.dWeights = inputT.dot(dOutput);

        Logger::backprop("| -> params.dWeights");
        Logger::debugTensor(LogLevel::BACKPROP, params.dWeights);

        params.dBiases = dOutput.sumColumns();
        Logger::backprop("| -> params.dBiases");
        Logger::debugTensor(LogLevel::BACKPROP, params.dBiases);

        // Calculate input gradients for backprop
        return dInput;
    }
};