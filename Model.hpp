#pragma once

#include <cstring>
#include <string_view>
#include <memory>

template <class T>
struct TensorView
{
    std::vector<T *> layers = nullptr;
    int            nb_layers  = 0;
};
