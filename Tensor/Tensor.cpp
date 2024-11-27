#include "Tensor.hpp"

// ----------------------------------------------------------- TRANSPOSE ----------------------------------------------------------- \\

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


// ----------------------------------------------------------- FILL UP WITH ZEROS ----------------------------------------------------------- \\

template <class T>
void fillZeroCPU(Tensor<T>& input) {
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            input[y][x] = 0;
        }
    }
}

// template definitions
template void fillZeroCPU(Tensor<float>& input);
template void fillZeroCPU(Tensor<double>& input);
template void fillZeroCPU(Tensor<int>& input);