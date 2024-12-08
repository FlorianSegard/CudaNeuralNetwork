#pragma once
#include "../Layers.hpp"
#include <iostream>
#include <typeinfo>

Tensor<float> backwardLinearGPU(Tensor<float> input);

Tensor<float> backwardLinearCPU(Tensor<float> input);


struct Linear : public Layer {
    LayerParams params;

    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
            : Layer(require_grad), params(inputSize, outputSize, device) {}

    Tensor<float> computeForward(Tensor<float>& input) override {
        // Linear forward: output = input @ weights.T + biases
        Tensor<float> weightsT = params.weights.transpose();

        // Dot matrix product [batch_size, input_size] @ [input_size, output_size]
        Tensor<float> output = input.dot(weightsT);

        // Add biases
        return output + params.biases;
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        // dInput = dOutput @ weights
        Tensor<float> dInput = dOutput.dot(params.weights);

        if (require_grad) {
            // dWeights = input.T @ dOutput
            Tensor<float> inputT = this->lastInput.transpose();
            params.dWeights = inputT.dot(dOutput);

            Tensor<float> a = Tensor<float>(1, dOutput.height, true);
            a.fillOnes();
            params.dBiases = dOutput.transpose().dot(a);
        }
        
        return dInput;
    }

    void setDevice(bool device) {
        params.switchDevice(device);
        onDeviceChanged(device);
    }
};