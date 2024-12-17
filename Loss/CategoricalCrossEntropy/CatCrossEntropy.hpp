#pragma once

#include "../../Tensor/Tensor.hpp"
#include "Logger/Logger.hpp"

float computeCatCrossEntropyLossGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients);

__global__ void crossCatEntropyLossKernel(const float* predictions, const float* targets,
                                      float* gradients, float* losses,
                                      int width, int height,
                                      size_t predStride, size_t targetStride, size_t gradStride);

float computeCatCrossEntropyLossCPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients);


inline std::pair<float, Tensor<float>> CategoricalCrossEntropyLoss(Tensor<float>& predictions, Tensor<float>& targets) {
    if (predictions.width != targets.width || predictions.height != targets.height) {
        throw std::invalid_argument("Dimensions do not match for loss calculation.");
    }
    Logger::loss(">>> CrossEntropyLoss");

    Tensor<float> grad(predictions.width, predictions.height, predictions.device);
    float loss;

    if (predictions.device) {
        loss = computeCatCrossEntropyLossGPU(predictions, targets, grad);
    } else {
        loss = computeCatCrossEntropyLossCPU(predictions, targets, grad);
    }

    Logger::loss("----- Final Grad -----");
    Logger::debugTensor(LogLevel::LOSS, grad);

    return {loss, std::move(grad)};
}
