#pragma once

#include "../Tensor/Tensor.hpp"

float sumOfSquaresCPU(const Tensor<float>& input);

__global__ void sumOfSquaresKernel(const float* input, int width, int height, size_t stride, float* partialSums);

float sumOfSquaresGPU(const Tensor<float>& input);

// A dummy Mean Squared Error (MSE) loss function
// Returns the loss value and computes dLoss/dOutput for backward.
inline std::pair<float, Tensor<float>> computeMSELoss(Tensor<float>& predictions, Tensor<float>& targets) {
    // Ensure dimensions match
    if (predictions.width != targets.width || predictions.height != targets.height) {
        throw std::invalid_argument("Dimensions do not match for loss calculation.");
    }

    Tensor<float> diff = predictions - targets;
    std::cout << "SUB LOSS ";
    diff.switchDevice(false).print();
    float sumSq = 0.0f;
    if (predictions.device) {
        sumSq = sumOfSquaresGPU(diff);
    } else {
        sumSq = sumOfSquaresCPU(diff);
    }

    // Compute MSE: loss = sum((pred - target)^2) / (N)
    const auto N = static_cast<float>(diff.width * diff.height);
    float loss = sumSq / N;

    // Compute gradient dLoss/dPred = 2*(pred - target)/N

    const float scale = 2.0f / N;
    Tensor<float> grad = diff * scale;
    std::cout << "loss = " << loss << ", GRAD ";
    grad.switchDevice(false).print();
    return {loss, std::move(grad)};
}
