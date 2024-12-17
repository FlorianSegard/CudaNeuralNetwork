#pragma once

#include <curand_kernel.h>

#include "../Layers.hpp"
#include "Logger/Logger.hpp"

void fillMaskGPU(Tensor<float>* mask, float drop_rate, curandState* states);

void fillMaskCPU(Tensor<float>* mask, float drop_rate);

void freeCurandStates();

void initializeCurandStates(curandState** d_states, int width, int height, size_t stride);

struct Dropout : public Layer {
    float drop_rate;
    Tensor<float> mask;
    bool eval_ = false;
    curandState* d_states = nullptr;

    Dropout(float drop_rate, int inputSize = 0, int outputSize = 0, bool device = false, bool require_grad = true)
            : Layer(inputSize, outputSize, device, require_grad), drop_rate(drop_rate) {
        if (drop_rate < 0.0f || drop_rate >= 1.0f) {
            throw std::runtime_error("Dropout rate must be in [0, 1)");
        }
    }

    ~Dropout() {
        if (d_states) {
            cudaFree(d_states);
            d_states = nullptr;
        }
    }

    Tensor<float> computeForward(Tensor<float>& input, bool eval) override {
        Logger::infer(">>> Dropout");
        eval_ = eval;

        if (eval) {
            return std::move(input);  // No scaling needed during inference
        }

        Tensor<float> local_mask(input.width, input.height, input.device);

        if (input.device) {
            if (!d_states) {
                initializeCurandStates(&d_states, input.width, input.height, input.stride);
            }
            fillMaskGPU(&local_mask, drop_rate, d_states);
        } else {
            fillMaskCPU(&local_mask, drop_rate);
        }

        mask = local_mask.clone();
        return input.termToTermMult(mask) * (1.0f / (1.0f - drop_rate));
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        Logger::backprop(">>> Dropout");
        if (!require_grad) return Tensor<float>();

        if (eval_)
            return std::move(dOutput);

        return dOutput.termToTermMult(mask) * (1.0f / (1.0f - drop_rate));
    }
};