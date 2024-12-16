#pragma once

#include <memory>
#include "../Layers/Layers.hpp"
#include "../Optimizer/Optimizer.hpp"
#include "../Layers/Linear/Linear.hpp"
#include <utility>
#include <vector>

#include "Logger/Logger.hpp"

float computeGradientNorm(const Tensor<float>& gradients) {
    float norm = 0.0f;
    for (int i = 0; i < gradients.height; i++) {
        for (int j = 0; j < gradients.width; j++) {
            norm += gradients[i][j] * gradients[i][j];
        }
    }
    return std::sqrt(norm);
}

struct Model
{
    std::vector<std::unique_ptr<Layer>> layers;
    SGD optimizer;
    bool test = true;

    Model() : optimizer(0.001f, 0) {}

    explicit Model(std::vector<std::unique_ptr<Layer>> other_layers)
            : layers(std::move(other_layers)), optimizer(0.001f, 0) {}

    Model(std::vector<std::unique_ptr<Layer>>& other_layers, SGD opt)
            : layers(std::move(other_layers)), optimizer(std::move(opt)) {}

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    void setOptimizer(const SGD& opt) {
        optimizer = opt;
    }

    Tensor<float> forward(Tensor<float> input) {
        for (auto& layer : layers) {
            input = layer->forward(input);

            Logger::infer("==== OUT of forward Layer ====");
            Logger::debugTensor(LogLevel::INFER, input);
        }
        return input;
    }

    void backward(Tensor<float>& dOutput) {

        Tensor<float> dInput = dOutput.clone();
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            Logger::backprop("==== IN of backward Layer ====");
            Logger::debugTensor(LogLevel::BACKPROP, dInput);

            dInput = (*it)->backward(dInput);
        }
    }

    void step() {
        for (auto& layer : layers) {
            if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
                if (test) {
                    Tensor<float> linear_param = linear->params.dWeights.switchDevice(false);
                    float norm = computeGradientNorm(linear_param);
                    if (norm > 100.0f) {
                        std::cout << "Gradient norm is exploding: " << norm << std::endl;
                        test = false;
                    }
                }

                optimizer.update(linear->params);
            }
        }
    }

    void switchDevice(bool device) {
        for (auto& layer : layers) {
            layer->switchDevice(device);
        }
    }

    void zeroGrad() {
        for (auto& layer : layers) {
            layer->zeroGrad();
        }
    }
};
