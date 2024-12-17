#pragma once
#include "../Optimizer/Optimizer.hpp"
#include <algorithm>

class Scheduler {
protected:
    Optimizer* optimizer;

public:
    explicit Scheduler(Optimizer* opt) : optimizer(opt) {}
    virtual ~Scheduler() = default;

    virtual void step(float currentLoss) = 0;
};



class ReduceLROnPlateau : public Scheduler {
private:
    float bestLoss;
    int patience;
    int counter;
    float factor;
    float min_lr;
    float threshold;

public:
    ReduceLROnPlateau(Optimizer* opt, int patience = 10, float factor = 0.1f, float min_lr = 1e-6f, float threshold = 1e-4f)
        : Scheduler(opt), bestLoss(std::numeric_limits<float>::infinity()), patience(patience), counter(0),
          factor(factor), min_lr(min_lr), threshold(threshold) {}

    void step(float currentLoss) {
        if (currentLoss < bestLoss - threshold) {
            bestLoss = currentLoss;
            counter = 0;
        }
        else {
            counter++;
            if (counter >= patience) {
                // std::cout << "lr before" << optimizer->getLearningRate() << std::endl;
                // std::cout << "lr reducing" << std::endl;

                float current_lr = optimizer->getLearningRate();
                float new_lr = std::max(current_lr * factor, min_lr);
                if (new_lr < current_lr) {
                    optimizer->setLearningRate(new_lr);
                }
                // std::cout << "lr after" << optimizer->getLearningRate() << std::endl;

                counter = 0;
            }
        }
    }
};
