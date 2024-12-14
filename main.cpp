#include <iostream>
#include <vector>
#include <random>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/Loss.hpp"
#include "Loader/ModelLoader.hpp"

const int TRAIN_SIZE = 5000;
const int TEST_SIZE = 1000;
const int BATCH_SIZE = 64;
const int INPUT_FEATURES = 3;
const int OUTPUT_FEATURES = 2;

void check_weights(Model* model) {
    std::cout << "------------- WEIGHTS MODEL LAYER 0 ------------- " << std::endl;  
    {
        model->switchDevice(false);

        Linear* linear1 = dynamic_cast<Linear*>(model->layers[0].get());
        if (linear1) {
            std::cout << "First layer weights after step:" << std::endl;
            linear1->params.weights.print();
        }
        model->switchDevice(true);
    }
    std::cout << "------------- WEIGHTS Gradient weights LAYER 0 ------------- " << std::endl;  
    {
        model->switchDevice(false);

        Linear* linear1 = dynamic_cast<Linear*>(model->layers[0].get());
        if (linear1) {
            std::cout << "First layer weights after step:" << std::endl;
            linear1->params.dWeights.print();
        }
        model->switchDevice(true);
    }
}

// Function to generate synthetic data
std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> generateDataset(int numSamples, int inputSize, int outputSize) {
    std::vector<Tensor<float>> inputs;
    std::vector<Tensor<float>> targets;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1); // Normal distribution with mean=0, std=1

    for (int i = 0; i < numSamples; i++) {
        Tensor<float> input(inputSize, 1, false);
        Tensor<float> target(outputSize, 1, false);

        for (int x = 0; x < inputSize; x++) {
            input[0][x] = static_cast<float>(d(gen));
        }
        for (int x = 0; x < outputSize; x++) {
            target[0][x] = static_cast<float>(d(gen));
        }

        inputs.push_back(std::move(input));
        targets.push_back(std::move(target));
    }

    return {std::move(inputs), std::move(targets)};
}

int main() {
    // Create a model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.01f));

    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, 4, onGPU));
    model.addLayer(std::make_unique<ReLU>());

    model.addLayer(std::make_unique<Linear>(4, OUTPUT_FEATURES, onGPU));

    // Generate training and testing datasets
    auto [trainInputs, trainTargets] = generateDataset(TRAIN_SIZE, INPUT_FEATURES, OUTPUT_FEATURES);
    auto [testInputs, testTargets] = generateDataset(TEST_SIZE, INPUT_FEATURES, OUTPUT_FEATURES);

    // Training loop
    for (int epoch = 0; epoch < 50; epoch++) {
        float totalLoss = 0.0f;

        for (size_t i = 0; i < trainInputs.size(); i += BATCH_SIZE) {
            // Prepare a batch
            size_t batchEnd = std::min(i + BATCH_SIZE, trainInputs.size());
            size_t batchSize = batchEnd - i;

            Tensor<float> inputBatch(INPUT_FEATURES, batchSize, false);
            Tensor<float> targetBatch(OUTPUT_FEATURES, batchSize, false);


            for (size_t b = 0; b < batchSize; b++) {
                for (int x = 0; x < INPUT_FEATURES; x++) {
                    inputBatch[b][x] = trainInputs[i + b][0][x];
                }
                for (int x = 0; x < OUTPUT_FEATURES; x++) {
                    targetBatch[b][x] = trainTargets[i + b][0][x];
                }
            }

            // inputBatch.print();
            // targetBatch.print();


            // Switch to GPU if applicable
            Tensor<float> inputGPU = inputBatch.switchDevice(onGPU);
            Tensor<float> targetGPU = targetBatch.switchDevice(onGPU);

            // Forward pass
            Tensor<float> predictions = model.forward(std::move(inputGPU));

            // Compute loss
            auto [loss, dOutput] = computeMSELoss(predictions, targetGPU);
            totalLoss += loss;

            // Backward pass
            model.backward(dOutput);
            // Update step
            model.step();

            // check_weights(&model);

            model.zeroGrad();

        }

        std::cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / TRAIN_SIZE << std::endl;
    }

    // Evaluate on test dataset
    float testLoss = 0.0f;

    for (size_t i = 0; i < testInputs.size(); i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, testInputs.size());
        size_t batchSize = batchEnd - i;

        Tensor<float> inputBatch(INPUT_FEATURES, batchSize, false);
        Tensor<float> targetBatch(OUTPUT_FEATURES, batchSize, false);

        for (size_t b = 0; b < batchSize; b++) {
            for (int x = 0; x < INPUT_FEATURES; x++) {
                inputBatch[b][x] = testInputs[i + b][0][x];
            }
            for (int x = 0; x < OUTPUT_FEATURES; x++) {
                targetBatch[b][x] = testTargets[i + b][0][x];
            }
        }

        Tensor<float> inputGPU = inputBatch.switchDevice(onGPU);
        Tensor<float> targetGPU = targetBatch.switchDevice(onGPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));

        auto [loss, _] = computeMSELoss(predictions, targetGPU);
        testLoss += loss;
    }

    std::cout << "Test Loss: " << testLoss / TEST_SIZE << std::endl;

    return 0;
}
