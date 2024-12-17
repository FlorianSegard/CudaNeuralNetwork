#include "BinCrossEntropy.hpp"

float computeBinaryCrossEntropyCPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float totalLoss = 0.0f;
    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - epsilon;

    int totalElements = predictions.width * predictions.height;

    for (int y = 0; y < predictions.height; y++) {
        for (int x = 0; x < predictions.width; x++) {
            // Clip predictions to prevent log(0)
            float pred = std::min(std::max(predictions[y][x], epsilon), one_minus_epsilon);
            float target = targets[y][x];

            // Compute binary cross entropy loss: -[t*log(p) + (1-t)*log(1-p)]
            totalLoss -= target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred);

            // Gradient computation: -(t/p - (1-t)/(1-p))
            // TODO: Simplified gradient for sigmoid + BCE: pred - target
            // -> gradients[y][x] = pred - target;
            gradients[y][x] = -(target / pred - (1.0f - target) / (1.0f - pred));
        }
    }

    return totalLoss / static_cast<float>(totalElements);
}
