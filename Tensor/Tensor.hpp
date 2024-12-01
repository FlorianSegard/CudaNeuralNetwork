#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <iostream>

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

// -------------------------------------------- DOT -------------------------------------------- \\

template <class T>
__global__ void dotGPUKernel(T* input, T* other, T* result, int width_input, int height_input, int width_output, size_t inputStride, size_t otherStride, size_t resultStride);

template <class T>
Tensor<T> dotGPU(const Tensor<T>& input, const Tensor<T>& other);

template <class T>
Tensor<T> dotCPU(const Tensor<T>& input, const Tensor<T>& other);

// -------------------------------------------- ADD -------------------------------------------- \\

template <class T>
__global__ void addKernel(const T* input, const T* other, T* result, int width, int height, int inputWidth, int inputHeight, int otherWidth, int otherHeight, size_t inputStride, size_t otherStride, size_t resultStride);

template <class T>
Tensor<T> addGPU(const Tensor<T>& input, const Tensor<T>& other);

template <class T>
Tensor<T> addCPU(const Tensor<T>& input, const Tensor<T>& other);

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


    Tensor clone() const;
    Tensor switchDevice(bool gpu);
    Tensor transpose() const;
    void fillZero();
    Tensor dot(const Tensor& other);
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
        return this->clone();
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
Tensor<T> Tensor<T>::dot(const Tensor<T>& other) {
    if (this->width != other.height)
        throw std::out_of_range("Matrix dimensions are incompatible for dot product.");
    if (this->device)
        return dotGPU(*this, other);
    else
        return dotCPU(*this, other);
}

//TODO implement random filling test later maybe: glorot filling