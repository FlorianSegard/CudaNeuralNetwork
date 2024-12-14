#include "ReLU.hpp"

// ----------------------------------------------------------- FORWARD ----------------------------------------------------------- \\

Tensor<float> reluCPU(Tensor<float>& input) {
    Tensor<float> result = input.clone();
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            result[y][x] = 0.0f;
            if (input[y][x] > 0.0f) {
                result[y][x] = input[y][x];
            }
        }
    }
    return result;
}

// ----------------------------------------------------------- BACKWARD ----------------------------------------------------------- \\

Tensor<float> reluBackwardCPU(Tensor<float>& input, Tensor<float>& dOutput) {
    Tensor<float> dInput = dOutput.clone();
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            dInput[y][x] = 0.0f;
            if (input[y][x] > 0.0f) 
            {
                dInput[y][x] = dOutput[y][x];
            }
        }
    }
    return dInput;
}