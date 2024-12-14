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

    Tensor<float> sub = predictions - targets;
    std::cout << "SUB LOSS ";
    sub.switchDevice(false).print();
    float sumSq = 0.0f;
    if (predictions.device) {
        sumSq = sumOfSquaresGPU(sub); 
    } else {
        sumSq = sumOfSquaresCPU(sub);
    }

    // Compute MSE: loss = sum((pred - target)^2) / (N)
    float loss = sumSq / (sub.width * sub.height);

    // Compute gradient dLoss/dPred = 2*(pred - target)/N

    float scale = 2.0f / (sub.width * sub.width);
    Tensor<float> grad = sub * scale;

    return {loss, std::move(grad)};
}
