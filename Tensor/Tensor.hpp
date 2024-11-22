#pragma once

#include <cstring>
#include <string_view>
#include <memory>

// View over a 2D buffer
template <class T>
struct TensorView
{
  T*             buffer = nullptr;
  int            width  = 0;
  int            height = 0;
  std::ptrdiff_t stride = 0;

    T* operator[](int y);
    const T* operator[](int y) const;
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

  Tensor clone() const;
};


template <class T>
Tensor<T>::Tensor(const char* path)
{
  int w, h, n;
  this->buffer = (T*)stbi_load(path, &w, &h, &n, sizeof(T));
  this->width  = w;
  this->height = h;
  this->stride = w * sizeof(T);
  this->deleter = stbi_image_free;
}

template <class T>
Tensor<T>::Tensor(int width, int height, bool device)
{
  static auto cudaDelete = [](void* ptr) { cudaFree(ptr); };
  this->width  = width;
  this->height = height;
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
    return (T *)((std::byte *)this->buffer + y * this->pitch);
  #else

  if (y < 0 || y >= height) {
    throw std::out_of_range("Row index out of range: " + std::to_string(y));
  }
  T* casted_buffer = (T *)((std::byte *)this->buffer + y * this->pitch);
  return casted_buffer;
}

template <class T>
__host__ __device__
const T* Tensor<T>::operator[](int y) const
{
  #ifdef __CUDA_ARCH__
    return (T *)((std::byte *)this->buffer + y * this->pitch);
  #else
  if (y < 0 || y >= height) {
    throw std::out_of_range("Row index out of range: " + std::to_string(y));
  }
  T* casted_buffer = (T *)((std::byte *)this->buffer + y * this->pitch);
  return casted_buffer;
}


template <class T>
Tensor<T> Tensor<T>::switchDevice(bool toDevice) const
{
    Tensor<T> result(this->width, this->height, toDevice);

    if (toDevice) {
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

    return result;
}

template <class T>
Tensor<T> Tensor<T>::transpose() const
{
    Tensor<T> result(this->height, this->width, false);

    for (int y = 0; y < this->height; ++y) {
        for (int x = 0; x < this->width; ++x) {
            result[x][y] = (*this)[y][x];
        }
    }

    return result;
}