#pragma once
#include "../Layers.hpp"
#include <iostream>
#include <typeinfo>


struct Linear : public Layer {


    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
            : Layer(inputSize, outputSize, device, require_grad){
                this->params.weights.fillOnes();
                this->params.biases.fillZero();
            }

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
        if (!require_grad) return Tensor<float>();

        Tensor<float> dInput = dOutput.dot(params.weights);

        // dWeights = input.T @ dOutput
        Tensor<float> inputT = this->lastInput.transpose();
        params.dWeights = inputT.dot(dOutput);
        params.dBiases = dOutput.clone();

        // Calculate input gradients for backprop
        return dOutput.dot(params.weights.transpose());
    }

    void setDevice(bool device) {
        params.switchDevice(device);
        onDeviceChanged(device);
    }
};