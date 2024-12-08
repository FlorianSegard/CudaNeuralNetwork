#include "Layers/Layers.hpp"

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(LayerParams& params) = 0;
};


class SGD : public Optimizer {
private:
    float learningRate;

public:
    explicit SGD(const float lr) : learningRate(lr) {}

    void update(LayerParams& params) override {
        params.weights = params.weights - params.dWeights * learningRate;
        params.biases = params.weights - params.dBiases * learningRate;
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
