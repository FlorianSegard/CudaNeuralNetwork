#pragma once

#include "../Tensor/Tensor.hpp"
#include "Logger/Logger.hpp"

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
    Logger::debug(">>> MSELoss");

    Tensor<float> diff = predictions - targets;

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
    Logger::debug("----- Final Grad -----");
    Logger::debugTensor(LogLevel::DEBUG, diff);

    return {loss, std::move(grad)};
}

float computeCrossEntropyLossGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients);

__global__ void crossEntropyLossKernel(const float* predictions, const float* targets,
                                      float* gradients, float* losses,
                                      int width, int height,
                                      size_t predStride, size_t targetStride, size_t gradStride);

float computeCrossEntropyLossCPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients);

// Main loss computation function
inline std::pair<float, Tensor<float>> computeCrossEntropyLoss(Tensor<float>& predictions, Tensor<float>& targets) {
    if (predictions.width != targets.width || predictions.height != targets.height) {
        throw std::invalid_argument("Dimensions do not match for loss calculation.");
    }
    Logger::debug(">>> CrossEntropyLoss");

    Tensor<float> grad(predictions.width, predictions.height, predictions.device);
    float loss;

    if (predictions.device) {
        loss = computeCrossEntropyLossGPU(predictions, targets, grad);
    } else {
        loss = computeCrossEntropyLossCPU(predictions, targets, grad);
    }

    Logger::debug("----- Final Grad -----");
    Logger::debugTensor(LogLevel::DEBUG, grad);

    return {loss, std::move(grad)};
}
