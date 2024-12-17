#include <iomanip>

#include "../Layers/Layers.hpp"
#include <unordered_map>

#include "Logger/Logger.hpp"

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(LayerParams& params) = 0;
};


class SGD : public Optimizer {
private:
    float learningRate;
    float clipValue;
    float momentum;
    float weightDecay;

    std::unordered_map<const LayerParams*, std::shared_ptr<Tensor<float>>> velocityWeightsMap;
    std::unordered_map<const LayerParams*, std::shared_ptr<Tensor<float>>> velocityBiasesMap;

public:
    explicit SGD(float lr = 0.001f, float momentum = 0.0f, float weightDecay = 0.0f, float clipValue = 0.0f)
        : learningRate(lr), momentum(momentum), weightDecay(weightDecay), clipValue(clipValue) {
    }

    // Getters
    float getLearningRate() const { return learningRate; }
    float getMomentum() const { return momentum; }
    float getWeightDecay() const { return weightDecay; }
    float getClipValue() const { return clipValue; }

    void update(LayerParams& params) override {
        if (weightDecay > 0.0f) {
            Tensor<float> weightDecayGrad = params.weights * weightDecay;
            params.dWeights = params.dWeights + weightDecayGrad;
        }

        if (clipValue > 0.0f) {
            params.dWeights.clipGradients(clipValue);
            params.dBiases.clipGradients(clipValue);
        }

        if (momentum == 0.0)
        {
            params.weights = params.weights - params.dWeights * learningRate;
            params.biases = params.biases - params.dBiases * learningRate;
        }
        else
        {
            initializeVelocities(params);

            auto& velocityWeights = velocityWeightsMap[&params];
            auto& velocityBiases = velocityBiasesMap[&params];

            *velocityWeights = (*velocityWeights) * momentum + params.dWeights * (1.0f - momentum);
            *velocityBiases = (*velocityBiases) * momentum + params.dBiases * (1.0f - momentum);

            params.weights = params.weights - (*velocityWeights) * learningRate;
            params.biases = params.biases - (*velocityBiases) * learningRate;
        }
    }


    void initializeVelocities(LayerParams& params) {
        if (velocityWeightsMap.find(&params) == velocityWeightsMap.end()) {
            velocityWeightsMap[&params] = std::make_shared<Tensor<float>>(
                params.weights.width, params.weights.height, params.weights.device);
            velocityWeightsMap[&params]->fillZero();

            velocityBiasesMap[&params] = std::make_shared<Tensor<float>>(
                params.biases.width, params.biases.height, params.biases.device);
            velocityBiasesMap[&params]->fillZero();
        }
    }

};


// class Adam : public Optimizer {
// private:
//     float learningRate;
//     float beta1;
//     float beta2;
//     float epsilon;
//     std::unordered_map<std::string, Tensor<float>> moments;

// public:
//     Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
//         : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

//     void update(LayerParams& params) override {
//     }
// };
