#pragma once

#include <memory>
#include "../Layers/Layers.hpp"
#include "../Optimizer/Optimizer.hpp"
#include "../Layers/Linear/Linear.hpp"
#include <utility>
#include <vector>

#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Sigmoid/Sigmoid.hpp"
#include "Layers/Softmax/Softmax.hpp"
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
    bool test_grad_explosion = false;

    Model() : optimizer(0.001f, 0) {}

    explicit Model(std::vector<std::unique_ptr<Layer>> other_layers)
            : layers(std::move(other_layers)), optimizer(0.001f, 0) {}

    Model(std::vector<std::unique_ptr<Layer>>& other_layers, SGD opt)
            : layers(std::move(other_layers)), optimizer(std::move(opt)) {}

    void addLayer(std::unique_ptr<Layer> layer) {
        std::string layerType;
        int inputDim = 0;
        int outputDim = 0;

        if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
            layerType = "Linear";
            inputDim = linear->params.inputSize;
            outputDim = linear->params.outputSize;
        } else if (dynamic_cast<ReLU*>(layer.get())) {
            layerType = "ReLU";
        } else if (dynamic_cast<Softmax*>(layer.get())) {
            layerType = "Softmax";
        } else if (dynamic_cast<Sigmoid*>(layer.get())) {
            layerType = "Softmax";
        }

        // Create visualization
        std::ostringstream ss;
        ss << (layers.size() == 0 ? "====== Model architecture ======\n" : "")
           << "Adding Layer (" << layers.size() + 1 << "):"
           << "\n ├── Type: " << layerType;

        if (inputDim > 0 && outputDim > 0) {
            ss << "\n ├── Input dim: " << inputDim
               << "\n └── Output dim: " << outputDim;
        } else {
            ss << "\n └── Activation layer";
        }

        Logger::debug(ss.str());

        layers.push_back(std::move(layer));
    }

    void setOptimizer(const SGD& opt) {
        optimizer = opt;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "====== SGD init ======"
           << "\n ├── learning rate: " << optimizer.getLearningRate()
           << "\n ├── momentum: " << (optimizer.getMomentum() != 0.0f? std::to_string(optimizer.getMomentum()) : "None")
           << "\n ├── weight_decay: " << (optimizer.getWeightDecay() != 0.0f ? std::to_string(optimizer.getWeightDecay()) : "None")
           << "\n └── grad clipping: " << (optimizer.getClipValue() != 0.0f ? std::to_string(optimizer.getClipValue()) : "None")
           << "\n";
        Logger::debug(ss.str());
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

        Tensor<float> dInput = std::move(dOutput);
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            Logger::backprop("==== IN of backward Layer ====");
            Logger::debugTensor(LogLevel::BACKPROP, dInput);

            dInput = (*it)->backward(dInput);
        }
    }

    void step() {
        for (auto& layer : layers) {
            if (auto* linear = dynamic_cast<Linear*>(layer.get())) {
                if (test_grad_explosion) {
                    Tensor<float> linear_param = linear->params.dWeights.switchDevice(false);
                    float norm = computeGradientNorm(linear_param);
                    if (norm > 100.0f) {
                        std::cout << "Gradient norm is exploding: " << norm << std::endl;
                        test_grad_explosion = false;
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
