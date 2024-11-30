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


// ----------------------------------------------------------- DOT ----------------------------------------------------------- \\

template <class T>
Tensor<T> dotCPU(const Tensor<T>& input, const Tensor<T>& other) {
    Tensor<T> result(other.width, input.height, false);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < other.width; ++x) {
            T sum = 0;
            for (int k = 0; k < input.width; ++k) {
                sum += input[y][k] * other[k][x];
            }
            result[y][x] = sum;
        }
    }
    return result;
}

// template definitions
template Tensor<float> dotCPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> dotCPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> dotCPU(const Tensor<int>& input, const Tensor<int>& other);