#include "CatCrossEntropy.hpp"
#include <cmath>

float computeCatCrossEntropyLossCPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float totalLoss = 0.0f;
    const float epsilon = 1e-7f;

    for (int y = 0; y < predictions.height; y++) {
        for (int x = 0; x < predictions.width; x++) {
            float pred = predictions[y][x];
            float target = targets[y][x];

            // gradient computation for softmax + cross-entropy
            gradients[y][x] = pred - target;

            if (target > 0) { // recommended by LLM -> only compute loss for positive targets
                totalLoss -= target * std::log(pred + epsilon);
            }
        }
    }

    return totalLoss / static_cast<float>(predictions.height * predictions.width);
}
