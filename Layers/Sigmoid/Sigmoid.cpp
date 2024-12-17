#include "Sigmoid.hpp"
#include <cmath>

Tensor<float> sigmoidCPU(Tensor<float>& input) {
    Tensor<float> output(input.width, input.height, false);

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            float val = input[y][x];
            output[y][x] = 1.0f / (1.0f + std::exp(-val));
        }
    }
    return output;
}

// gradient = sigmoid(x) * (1 - sigmoid(x)) * dOutput
Tensor<float> sigmoidBackwardCPU(Tensor<float>& output, Tensor<float>& dOutput) {
    Tensor<float> dInput(output.width, output.height, false);

    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            float sigmoid_x = output[y][x];
            dInput[y][x] = sigmoid_x * (1.0f - sigmoid_x) * dOutput[y][x];
        }
    }
    return dInput;
}