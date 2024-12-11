#include "Loss.hpp"

float sumOfSquaresCPU(const Tensor<float>& input) {
    float sum = 0.0f;
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float val = input[y][x];
            sum += val * val;
        }
    }
    return sum;
}
