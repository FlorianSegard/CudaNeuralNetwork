#pragma once

#include <cstring>
#include <string_view>
#include <memory>
#include "../Layers/Layers.hpp"

struct Model
{
    std::vector<Layer> layers;

    Model() : layers() {}
    Model(const std::vector<Layer>& other_layers) : layers(other_layers) {}

    void addLayer(Layer layer_toadd)
    {
        layers.push_back(layer_toadd);
    }

    Tensor<float> forward(Tensor<float> input) 
    {
        for (auto& layer : layers) {
            input = layer.forward(input);
        }
        return input;
    }

    void backward(Tensor<float> output)
    {
        Tensor<float> dInput = dOutput;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            dInput = it->backward(dInput);
        }
    }
};


// struct LayerParams {
//     Tensor<float> weights;
//     Tensor<float> biases;
//     Tensor<float> dWeights;   // Gradients for backward
//     Tensor<float> dBiases;    // Gradients for backward
//     int inputSize = 0;
//     int outputSize = 0;

//     LayerParams() = default; // Default constructor
//     LayerParams(int inputSize, int outputSize, bool device)
//         : weights(inputSize, outputSize, device),
//           biases(1, outputSize, device),
//           dWeights(inputSize, outputSize, device),
//           dBiases(1, outputSize, device),
//           inputSize(inputSize), outputSize(outputSize) {}


//     // TODO in Tensor.hpp
//     void switchDevice(bool device) {
//         // Logic to reallocate or transfer data between CPU and GPU
//         weights.switchDevice(device);
//         biases.switchDevice(device);
//         dWeights.switchDevice(device);
//         dBiases.switchDevice(device);
//     }

// };
