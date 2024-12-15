#include "Tensor.hpp"

#include <cmath>
#include <bits/random.h>

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

// ----------------------------------------------------------- TERM TO TERM MULT ----------------------------------------------------------- \\

template <class T>
Tensor<T> termtotermMultCPU(const Tensor<T>& input, const Tensor<T>& other) {
    Tensor<T> result(input.width, input.height, false);

    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            int inputIndex = x + (input.stride / sizeof(T)) * y;
            int otherIndex = x + (other.stride / sizeof(T)) * y;
            int resultIndex = x + (result.stride / sizeof(T)) * y;

            result.buffer[resultIndex] = input.buffer[inputIndex] * other.buffer[otherIndex];
        }
    }

    return result;
}

// template definitions
template Tensor<float> termtotermMultCPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> termtotermMultCPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> termtotermMultCPU(const Tensor<int>& input, const Tensor<int>& other);


// ----------------------------------------------------------- ADD ----------------------------------------------------------- \\

template <class T>
Tensor<T> addCPU(const Tensor<T>& input, const Tensor<T>& other) {
    int resultWidth = std::max(input.width, other.width); 
    int resultHeight = std::max(input.height, other.height);

    Tensor<T> result(resultWidth, resultHeight, false);

    for (int y = 0; y < resultHeight; y++) {
        for (int x = 0; x < resultWidth; x++) {
            int inputIndex = (x % input.width) + (input.stride / sizeof(T)) * (y % input.height);
            int otherIndex = (x % other.width) + (other.stride / sizeof(T)) * (y % other.height);
            int resultIndex = x + (result.stride / sizeof(T)) * y;

            result.buffer[resultIndex] = input.buffer[inputIndex] + other.buffer[otherIndex];
        }
    }

    return result;
}

// template definitions
template Tensor<float> addCPU(const Tensor<float>& input, const Tensor<float>& other);
template Tensor<double> addCPU(const Tensor<double>& input, const Tensor<double>& other);
template Tensor<int> addCPU(const Tensor<int>& input, const Tensor<int>& other);

// ----------------------------------------------------------- Scalar Mult ----------------------------------------------------------- \\

template <class T>
Tensor<T> scalarMultiplyCPU(const Tensor<T>& input, const T scalar) {
    Tensor<T> result(input.width, input.height, false);

    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            int inputIndex = (x % input.width) + (input.stride / sizeof(T)) * (y % input.height);

            result.buffer[inputIndex] = input.buffer[inputIndex] * scalar;
        }
    }
    return result;
}

// template definitions
template Tensor<float> scalarMultiplyCPU(const Tensor<float>& input, const float other);
template Tensor<double> scalarMultiplyCPU(const Tensor<double>& input, const double other);
template Tensor<int> scalarMultiplyCPU(const Tensor<int>& input, const int other);

// ----------------------------------------------------------- FILE ONES ----------------------------------------------------------- \\

template <class T>
void fillOnesCPU(Tensor<T>& input) {
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            input[y][x] = T(1);
        }
    }
}

// template definitions
template void fillOnesCPU(Tensor<float>& input);
template void fillOnesCPU(Tensor<double>& input);
template void fillOnesCPU(Tensor<int>& input);

// ----------------------------------------------------------- Clip Gradients ----------------------------------------------------------- \\

template <class T>
void clipGradientsCPU(Tensor<T>& gradients, const T clipValue) {
    for (int i = 0; i < gradients.height; i++) {
        for (int j = 0; j < gradients.width; j++) {
            gradients[i][j] = std::max(std::min(gradients[i][j], clipValue), -clipValue);
        }
    }
}

template void clipGradientsCPU(Tensor<float>& input, float clipValue);
template void clipGradientsCPU(Tensor<double>& input, double clipValue);
template void clipGradientsCPU(Tensor<int>& input, int clipValue);

// ----------------------------------------------------------- Xavier Init weight ----------------------------------------------------------- \\

template <class T>
void initializeWeightsCPU(Tensor<T>& weights, float limit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    for (int i = 0; i < weights.height; i++) {
        for (int j = 0; j < weights.width; j++) {
            weights[i][j] = dis(gen);
        }
    }
}

template void initializeWeightsCPU(Tensor<float>& weights, float limit);
template void initializeWeightsCPU(Tensor<double>& weights, float limit);