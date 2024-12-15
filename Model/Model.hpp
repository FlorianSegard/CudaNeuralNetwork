#pragma once

#include <memory>
#include "../Layers/Layers.hpp"
#include "../Optimizer/Optimizer.hpp"
#include "../Layers/Linear/Linear.hpp"
#include <utility>
#include <vector>

#include "Logger/Logger.hpp"


struct Model
{
    std::vector<std::unique_ptr<Layer>> layers;
    SGD optimizer;

    Model() : optimizer(0.001f) {}

    explicit Model(std::vector<std::unique_ptr<Layer>>& other_layers)
            : layers(std::move(other_layers)), optimizer(0.001f) {}

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
            if (auto* linear = dynamic_cast<Linear*>(it->get())) {
                // Log gradient norms before backward
                float gradNorm = 0.0f;
                for (int i = 0; i < dInput.height; i++) {
                    for (int j = 0; j < dInput.width; j++) {
                        gradNorm += dInput[i][j] * dInput[i][j];
                    }
                }
                Logger::debug("Gradient norm: " + std::to_string(std::sqrt(gradNorm)));
            }
            dInput = (*it)->backward(dInput);
        }
    }

    void step() {
        for (auto& layer : layers) {
            if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
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
