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
  this->deleter = stbi_Tensor_free;
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
T* Tensor<T>::operator[](int y) noexcept
{
  T* casted_buffer = (T *)((std::byte *)this->buffer + y * this->pitch);
  return casted_buffer;
}

/*
template <class T>
Tensor<T> Tensor<T>::clone() const
{
  Tensor<T> out(this->width, this->height);
  for (int y = 0; y < this->height; ++y)
    memcpy((char*)out.buffer + y * out.stride, //
                (char*)this->buffer + y * this->stride, //
                this->width * sizeof(T));
  return out;
}
*/