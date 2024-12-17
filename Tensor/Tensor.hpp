#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <iostream>
#include <random>


template <class T>
struct Tensor;

// -------------------------------------------- TRANSPOSE -------------------------------------------- \\

template <class T>
__global__ void transposeKernel(const T* input, T* output, int width, int height, size_t inStride, size_t outStride);

template <class T>
Tensor<T> transposeGPU(const Tensor<T>& input);

template <class T>
Tensor<T> transposeCPU(const Tensor<T>& input);

// -------------------------------------------- FILL UP WITH ZEROS -------------------------------------------- \\

template <class T>
__global__ void fillZeroKernel(T* input, int width, int height, size_t inStride);

template <class T>
void fillZeroGPU(Tensor<T>& input);

template <class T>
void fillZeroCPU(Tensor<T>& input);

// -------------------------------------------- FILL UP WITH ONES -------------------------------------------- \\

template <class T>
__global__ void fillOnesKernel(T* input, int width, int height, size_t inStride);

template <class T>
void fillOnesGPU(Tensor<T>& input);

template <class T>
void fillOnesCPU(Tensor<T>& input);

// -------------------------------------------- DOT -------------------------------------------- \\

template <class T>
__global__ void dotGPUKernel(T* input, T* other, T* result, int width_input, int height_input, int width_output, size_t inputStride, size_t otherStride, size_t resultStride);

template <class T>
Tensor<T> dotGPU(const Tensor<T>& input, const Tensor<T>& other);

template <class T>
Tensor<T> dotCPU(const Tensor<T>& input, const Tensor<T>& other);


// -------------------------------------------- TERM TO TERM MULT -------------------------------------------- \\

template <class T>
__global__ void termtotermMultKernel(const T* input, const T* other, T* result, int width, int height, size_t inputStride, size_t otherStride, size_t resultStride);

template <class T>
Tensor<T> termtotermMultGPU(const Tensor<T>& input, const Tensor<T>& other);

template <class T>
Tensor<T> termtotermMultCPU(const Tensor<T>& input, const Tensor<T>& other);

    
// -------------------------------------------- ADD -------------------------------------------- \\

template <class T>
__global__ void addKernel(const T* input, const T* other, T* result, int width, int height, int inputWidth, int inputHeight, int otherWidth, int otherHeight, size_t inputStride, size_t otherStride, size_t resultStride);

template <class T>
Tensor<T> addGPU(const Tensor<T>& input, const Tensor<T>& other);

template <class T>
Tensor<T> addCPU(const Tensor<T>& input, const Tensor<T>& other);

// -------------------------------------------- SUB -------------------------------------------- \\

template <class T>
__global__ void scalarMultiplyKernel(const T* input, T* output, T scalar, int width, int height, size_t inStride, size_t outStride);

template <class T>
Tensor<T> scalarMultiplyGPU(const Tensor<T>& input, T scalar);

template <class T>
Tensor<T> scalarMultiplyCPU(const Tensor<T>& input, T scalar);

// ----------------------------------------------------------- Clip Gradients ----------------------------------------------------------- \\

template <class T>
__global__ void clipGradientsKernel(T* gradients, int width, int height, size_t stride, T clipValue);

template <class T>
void clipGradientsGPU(Tensor<T>& gradients, T clipValue);

template <class T>
void clipGradientsCPU(Tensor<T>& gradients, T clipValue);

// ----------------------------------------------------------- Xavier Init weight ----------------------------------------------------------- \\

template <class T>
__global__ void initWeightsKernel(T* weights, int width, int height, size_t stride, float limit, unsigned int seed);

template <class T>
void initWeightsGPU(Tensor<T>& weights, float limit);

template <class T>
void initializeWeightsCPU(Tensor<T>& weights, float limit);

// ----------------------------------------------------------- Sum column ----------------------------------------------------------- \\

template <class T>
__global__ void sumColumnsKernel(const T* input, T* output, int width, int height, size_t stride);

template <class T>
Tensor<T> sumColumnsGPU(Tensor<T>& input);

template <class T>
Tensor<T> sumColumnsCPU(Tensor<T>& input);

// -------------------------------------------- DEFINITIONS -------------------------------------------- \\

// View over a 2D buffer
template <class T>
struct TensorView {
    T* buffer = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride = 0;
    bool device = false;
};

// Class that owns data
template <class T>
struct Tensor : TensorView<T> {
    void (*deleter)(void*) = nullptr;

    Tensor() = default;
    Tensor(int width, int height, bool device = false);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;


    T* operator[](int y);
    const T* operator[](int y) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(float scalar) const;

    Tensor clone() const;
    Tensor switchDevice(bool gpu);
    Tensor transpose() const;
    Tensor dot(const Tensor& other);
    Tensor termToTermMult(const Tensor& other);
    Tensor sumColumns();
    void print();
    void fillZero();
    void fillOnes();
    void clipGradients(T clipValue);
    void initializeWeights(int fanIn, int fanOut);
};

template <class T>
Tensor<T>::Tensor(int width, int height, bool device) {
    static auto cudaDelete = [](void* ptr) { cudaFree(ptr); };
    this->width = width;
    this->height = height;
    this->device = device;

    if (device) {
        size_t pitch;
        cudaMallocPitch((void**)&this->buffer, &pitch, this->width * sizeof(T), this->height);
        this->stride = pitch;
        this->deleter = cudaDelete;
    } else {
        this->stride = width * sizeof(T);
        this->buffer = (T*)malloc(this->height * this->stride);
        this->deleter = free;
    }
}

template <class T>
Tensor<T>::~Tensor() {
    if (this->buffer && this->deleter) {
        this->deleter(this->buffer);
    }
    this->buffer = nullptr;
}

template <class T>
Tensor<T>::Tensor(Tensor&& other) noexcept {
    std::swap((TensorView<T>&)(*this), (TensorView<T>&)(other));
}

template <class T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    std::swap((TensorView<T>&)(*this), (TensorView<T>&)(other));
    return *this;
}

template <class T>
T* Tensor<T>::operator[](int y) {
    if (this->device)
        return (T*)((std::byte*)this->buffer + y * this->stride);

    if (y < 0 || y >= this->height) {
        throw std::out_of_range("Row index out of range: " + std::to_string(y));
    }
    return (T*)((std::byte*)this->buffer + y * this->stride);
}

template <class T>
const T* Tensor<T>::operator[](int y) const {
    if (this->device)
        return (T*)((std::byte*)this->buffer + y * this->stride);

    if (y < 0 || y >= this->height) {
        throw std::out_of_range("Row index out of range: " + std::to_string(y));
    }
    return (T*)((std::byte*)this->buffer + y * this->stride);
}

template <class T>
Tensor<T> Tensor<T>::operator*(float scalar) const {
    if (this->device)
        return scalarMultiplyGPU(*this, scalar);
    else
        return scalarMultiplyCPU(*this, scalar);
}

// template <class T>
// Tensor<T> Tensor<T>::operator+(Tensor<T>& other) {
//     if ((this->width != other.width && this->width != 1 && other.width != 1) ||
//         (this->height != other.height && this->height != 1 && other.height != 1)) {
//         throw std::invalid_argument("Tensors are not broadcast-compatible for addition.");
//     }


//     if (this->device)
//         return addGPU(*this, other);
//     else
//         return addCPU(*this, other);
// }

template <class T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    if ((this->width != other.width && this->width != 1 && other.width != 1) || // to take into account the diffusion if one of the height or width is 1
        (this->height != other.height && this->height != 1 && other.height != 1)) {
        throw std::invalid_argument("Tensors are not broadcast-compatible for addition.");
    }

    if (this->device)
        return addGPU(*this, other);
    else
        return addCPU(*this, other);
}

template <class T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    return *this + (other * -1.0f);
}

template <class T>
Tensor<T> Tensor<T>::clone() const {
    Tensor<T> result(this->width, this->height, this->device);

    if (this->device) {
        cudaMemcpy2D(result.buffer, result.stride, this->buffer, this->stride,
                     this->width * sizeof(T), this->height, cudaMemcpyDeviceToDevice);
    } else {
        std::memcpy(result.buffer, this->buffer, this->height * this->stride);
    }

    return result;
}

// true if wants to go to gpu false otherwise.
template <class T>
Tensor<T> Tensor<T>::switchDevice(bool gpu) {
    if (gpu == this->device) {
        return Tensor<T>(std::move(*this));
    }

    Tensor<T> result(this->width, this->height, gpu);

    if (gpu) {
        cudaMemcpy2D(result.buffer, result.stride, this->buffer, this->stride,
                     this->width * sizeof(T), this->height, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy2D(result.buffer, result.stride, this->buffer, this->stride,
                     this->width * sizeof(T), this->height, cudaMemcpyDeviceToHost);
    }

    return result;
}

// if transpose in place be careful to change the stride
template <class T>
Tensor<T> Tensor<T>::transpose() const {
    if (this->device)
        return transposeGPU(*this);
    else
        return transposeCPU(*this);
}

template <class T>
void Tensor<T>::fillZero() {
    if (this->device)
        fillZeroGPU(*this);
    else
        fillZeroCPU(*this);
}

template <class T>
void Tensor<T>::fillOnes() {
    if (this->device)
        fillOnesGPU(*this);
    else
        fillOnesCPU(*this);
}

template <class T>
Tensor<T> Tensor<T>::dot(const Tensor<T>& other) {
    if (this->width != other.height)
        throw std::out_of_range("Matrix dimensions are incompatible for dot product.");

    if (this->device)
        return dotGPU(*this, other);
    else
        return dotCPU(*this, other);
}

template <class T>
Tensor<T> Tensor<T>::termToTermMult(const Tensor<T>& other) {
    if (this->width != other.width || this->height != other.height)
        throw std::out_of_range("Matrix dimensions are incompatible for term to term product.");

    if (this->device)
        return termtotermMultGPU(*this, other);
    else
        return termtotermMultCPU(*this, other);
}


template <class T>
void Tensor<T>::print() {
    if (this->device) {
        throw std::runtime_error("Tensor is on GPU; switch to CPU before printing.");
    }

    std::cout << "Tensor (CPU): " << this->height << "x" << this->width << std::endl;
    for (int y = 0; y < this->height; y++) {
        for (int x = 0; x < this->width; x++) {
            std::cout << (*this)[y][x] << " ";
        }
        std::cout << std::endl;
    }
}

template <class T>
void Tensor<T>::clipGradients(const T clipValue) {
    if (this->device)
        clipGradientsGPU(*this, clipValue);
    else
        clipGradientsCPU(*this, clipValue);
}

template <class T>
void Tensor<T>::initializeWeights(int fanIn, int fanOut) {
    float limit = std::sqrt(2.0f / (float)(fanIn + fanOut));

    if (this->device)
        initWeightsGPU(*this, limit);
    else
        initializeWeightsCPU(*this, limit);
}


template <class T>
Tensor<T> Tensor<T>::sumColumns() {
    if (this->width == 0 || this->height == 0) {
        throw std::invalid_argument("Tensor dimensions must be non-zero.");
    }

    if (this->device) {
        return sumColumnsGPU(*this);
    } else {
        return sumColumnsCPU(*this);
    }
}


//TODO implement random filling test later maybe: glorot filling