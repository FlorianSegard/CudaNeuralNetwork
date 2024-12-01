#pragma once

#include <cstring>
#include <string_view>
#include <memory>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// View over a 2D buffer
template <class T>
struct TensorView
{
  T*             buffer = nullptr;
  int            width  = 0;
  int            height = 0;
  std::ptrdiff_t stride = 0;
  bool device = false;

};

// Class that owns data
template <class T>
struct Tensor : TensorView<T>
{
  void (*deleter) (void*) = nullptr;

  Tensor() = default;
  Tensor(int width, int height, bool device = false);
  Tensor(const char* path);
  Tensor(std::string_view path) : Tensor(path.data()) {} 
  ~Tensor();

  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;


  __host__ __device__ T* operator[](int y);
  __host__ __device__ const T* operator[](int y) const;

  Tensor clone() const;
  Tensor switchDevice(bool gpu);
  Tensor transpose() const;
};


// template <class T>
// Tensor<T>::Tensor(const char* path)
// {
//   int w, h, n;
//   this->buffer = (T*)stbi_load(path, &w, &h, &n, sizeof(T));
//   this->width  = w;
//   this->height = h;
//   this->stride = w * sizeof(T);
//   this->deleter = stbi_image_free;
//   this->device = false;
// }

template <class T>
Tensor<T>::Tensor(int width, int height, bool device)
{
  static auto cudaDelete = [](void* ptr) { cudaFree(ptr); };
  this->width  = width;
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
Tensor<T>::~Tensor()
{
  if (this->buffer && this->deleter)
    this->deleter(this->buffer);
  this->buffer = nullptr;
}

template <class T>
Tensor<T>::Tensor(Tensor&& other) noexcept
{
  std::swap((TensorView<T>&)(*this), (TensorView<T>&)(other));
}

template <class T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept
{
  std::swap((TensorView<T>&)(*this), (TensorView<T>&)(other));
  return *this;
}

template <class T>
__host__ __device__
T* Tensor<T>::operator[](int y)
{
  #ifdef __CUDA_ARCH__
    return (T *)((std::byte *)this->buffer + y * this->stride);
  #else

  if (y < 0 || y >= this->height) {
    throw std::out_of_range("Row index out of range: " + std::to_string(y));
  }
  T* casted_buffer = (T *)((std::byte *)this->buffer + y * this->stride);
  return casted_buffer;
  #endif
}

template <class T>
__host__ __device__
const T* Tensor<T>::operator[](int y) const
{
  #ifdef __CUDA_ARCH__
    return (T *)((std::byte *)this->buffer + y * this->stride);
  #else
  if (y < 0 || y >= this->height) {
    throw std::out_of_range("Row index out of range: " + std::to_string(y));
  }
  T* casted_buffer = (T *)((std::byte *)this->buffer + y * this->stride);
  return casted_buffer;
  #endif
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
Tensor<T> Tensor<T>::switchDevice(bool gpu)
{
    if (gpu == this->device) {
      return this->clone();
    }
    Tensor<T> result(this->width, this->height, gpu);

    if (gpu) {
        cudaMemcpy2D(
            result.buffer, result.stride,
            this->buffer, this->stride,
            this->width * sizeof(T), this->height,
            cudaMemcpyHostToDevice
        );
    } else {
        cudaMemcpy2D(
            result.buffer, result.stride,
            this->buffer, this->stride,
            this->width * sizeof(T), this->height,
            cudaMemcpyDeviceToHost
        );
    }
    this->device = gpu;

    return result;
}

#include "Tensor.cu"
#include "Tensor.cpp"


template <class T>
Tensor<T> Tensor<T>::transpose() const
{
    if (this->device)
    {
      return transposeGPU(*this);
    }
    else
    {
      return transposeCPU(*this);
    }
}