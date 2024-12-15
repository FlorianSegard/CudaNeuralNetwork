#pragma once
#include "../Layers.hpp"
#include <iostream>
#include <random>
#include <typeinfo>

#include "Logger/Logger.hpp"

void initializeWeights(Tensor<float>& weights, int fanIn, int fanOut) {
    // Xavier/Glorot initialization
    float limit = std::sqrt(2.0f / (float) (fanIn + fanOut));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    if (weights.device) {
        // If on GPU, initialize on CPU first then transfer
        Tensor<float> cpuWeights(weights.width, weights.height, false);
        for (int i = 0; i < weights.height; i++) {
            for (int j = 0; j < weights.width; j++) {
                cpuWeights[i][j] = dis(gen);
            }
        }
        weights = cpuWeights.switchDevice(true);
    } else {
        for (int i = 0; i < weights.height; i++) {
            for (int j = 0; j < weights.width; j++) {
                weights[i][j] = dis(gen);
            }
        }
    }
}


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
        // Logger::backprop(">>> Linear");
        if (!require_grad) return Tensor<float>();

        Tensor<float> dInput = dOutput.dot(params.weights.transpose());
        // Logger::backprop("| -> dInput");
        // Logger::debugTensor(LogLevel::BACKPROP, dInput);

        // dWeights = input.T @ dOutput

        Tensor<float> inputT = this->lastInput.transpose();
        params.dWeights = inputT.dot(dOutput);

        // Logger::backprop("| -> params.dWeights");
        // Logger::debugTensor(LogLevel::BACKPROP, params.dWeights);

        params.dBiases = dOutput.sumColumns();
        // Calculate input gradients for backprop
        return dInput;
    }
};