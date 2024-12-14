#include "Softmax.hpp"
#include <cmath>
#include <cfloat>

// Forward Pass on CPU
Tensor<float> softmaxCPU(Tensor<float>& input) {
    Tensor<float> output(input.width, input.height, false);

    for (int y = 0; y < input.height; y++) {
        float maxVal = -FLT_MAX;
        float sum = 0.0f;

        for (int x = 0; x < input.width; x++) {
            maxVal = std::max(maxVal, input[y][x]);
        }

        for (int x = 0; x < input.width; x++) {
            output[y][x] = std::exp(input[y][x] - maxVal);
            sum += output[y][x];
        }

        for (int x = 0; x < input.width; x++) {
            output[y][x] /= sum;
        }
    }
    return output;
}

// Backward Pass on CPU
Tensor<float> softmaxBackwardCPU(Tensor<float>& output, Tensor<float>& dOutput) {
    Tensor<float> dInput(output.width, output.height, false);

    for (int y = 0; y < output.height; y++) {
        for (int i = 0; i < output.width; i++) {
            float gradient = 0.0f;

            for (int j = 0; j < output.width; j++) {
                float delta = (i == j) ? output[y][i] * (1 - output[y][j])
                                       : -output[y][i] * output[y][j];
                gradient += delta * dOutput[y][j];
            }

            dInput[y][i] = gradient;
        }
    }
    return dInput;
}
