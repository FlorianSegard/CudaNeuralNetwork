#include "Tensor.hpp"

template <class T>
Tensor<T> transposeCPU(const Tensor<T>& input) {
    Tensor<T> result(input.height, input.width, false);

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            result[x][y] = input[y][x];
        }
    }

    return result;
}

// template definitions
template Tensor<float> transposeCPU(const Tensor<float>& input);
template Tensor<double> transposeCPU(const Tensor<double>& input);
template Tensor<int> transposeCPU(const Tensor<int>& input);