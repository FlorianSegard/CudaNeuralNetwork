#include "../Layers/Layers.hpp"
<<<<<<< HEAD
#include <unordered_map>

=======
#include "../Tensor/Tensor.hpp"
>>>>>>> 98670720f6cb58523184f1fec1153a6fffb1d198

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(LayerParams& params) = 0;
};


class SGD : public Optimizer {
private:
    float learningRate;
    float clipValue;

    std::unordered_map<const LayerParams*, std::shared_ptr<Tensor<float>>> velocityWeightsMap;
    std::unordered_map<const LayerParams*, std::shared_ptr<Tensor<float>>> velocityBiasesMap;
public:

    float momentum = 0;

    explicit SGD(float lr, float momentum) : learningRate(lr), momentum(momentum) {}

    void update(LayerParams& params) override {

        initializeVelocities(params);

        if (momentum == 0)
        {
            params.weights = params.weights - params.dWeights * learningRate;
            params.biases = params.biases - params.dBiases * learningRate;
        }
        else
        {
            auto& velocityWeights = velocityWeightsMap[&params];
            auto& velocityBiases = velocityBiasesMap[&params];
            // std::cout << "3" << std::endl;
            if (!velocityWeights || !velocityBiases) {
                throw std::runtime_error("Velocity tensors are not properly initialized.");
            }
            *velocityWeights = (*velocityWeights) * momentum + params.dWeights * (1.0f - momentum);
            // std::cout << "4" << std::endl;

            *velocityBiases = (*velocityBiases) * momentum + params.dBiases * (1.0f - momentum);
            // std::cout << "5" << std::endl;

            params.weights = params.weights - (*velocityWeights) * learningRate;
            params.biases = params.biases - (*velocityBiases) * learningRate;
            // std::cout << "6" << std::endl;

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
