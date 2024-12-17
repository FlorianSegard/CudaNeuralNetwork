#include "Dropout.hpp"
#include <random>

void fillMaskCPU(Tensor<float>* mask, float drop_rate)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(1.0f - drop_rate);

    for (int y = 0; y < mask->height; y++) {
        for (int x = 0; x < mask->width; x++) {
            (*mask)[y][x] = d(gen) ? 1.0f : 0.0f;
        }
    }
}


