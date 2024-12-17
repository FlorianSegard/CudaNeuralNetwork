#pragma once

#include <cfloat>

#include "../Layers.hpp"
#include "Logger/Logger.hpp"

void fillMaskGPU(Tensor<float>* mask, float drop_rate);

void fillMaskCPU(Tensor<float>* mask, float drop_rate);

void freeCurandStates();

struct Dropout : public Layer {
    float drop_rate;
    Tensor<float> mask;
    bool eval_ = false;
    
    Dropout(float drop_rate, int inputSize = 0, int outputSize = 0, bool device = false, bool require_grad = true)
            : Layer(inputSize, outputSize, device, require_grad), drop_rate(drop_rate) {
    }

    Tensor<float> computeForward(Tensor<float>& input, bool eval) override {
        Logger::infer(">>> Dropout");
        eval_ = eval;
        if (eval) {
            return std::move(input);
        }
        Tensor<float> local_mask(input.width, input.height, input.device);

        if (input.device)
            fillMaskGPU(&local_mask, drop_rate);
        else
            fillMaskCPU(&local_mask, drop_rate);
        Tensor<float> output = input.termToTermMult(local_mask) * (1.0f / (1.0f - drop_rate));

        mask = local_mask.clone();

        return output;
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        // dInput = dOutput @ weights
        Logger::backprop(">>> Dropout");
        if (!require_grad) return Tensor<float>();

        if (eval_)
            return std::move(dOutput);

        Tensor<float> dInput = dOutput.termToTermMult(mask);

        return dInput;
    }
};