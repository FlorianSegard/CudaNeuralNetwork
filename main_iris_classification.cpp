// main3.cpp
#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Softmax/Softmax.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/CategoricalCrossEntropy/CatCrossEntropy.hpp"
#include "Loss/MeanSquaredError/Mse.hpp"
#include "Logger/Logger.hpp"
#include "Loader/TabularLoader.hpp"

void train_iris() {
    std::cout << "\n=== Training Iris Classification Model ===\n" << std::endl;

    // training data
    auto [train_features, train_labels] = TabularLoader::loadCSV(
        "/home/alex/CudaNeuralNetwork/onnx_generator/data/Iris/train_data.csv",  // Path to your training data
        ',',              // delimiter
        false,            // don't normalize (data is already normalized)
        -1,               // load all rows
        true              // skip header
    );

    // test data
    auto [test_features, test_labels] = TabularLoader::loadCSV(
        "/home/alex/CudaNeuralNetwork/onnx_generator/data/Iris/test_data.csv",   // Path to your test data
        ',',
        false,
        -1,
        true
    );

    // Create model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.01f)); // Learning rate 0.01, momentum 0.9

    // Architecture for Iris classification
    // 4 input features -> 8 hidden -> 3 classes
    model.addLayer(std::make_unique<Linear>(4, 8, onGPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(8, 3, onGPU));
    model.addLayer(std::make_unique<Softmax>(true));

    // Training loop
    const int EPOCHS = 100;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        float correct = 0.0f;

        // Process all training samples
        Tensor<float> batchFeatures(4, train_features.size(), false);
        Tensor<float> batchLabels(3, train_features.size(), false);

        // Prepare batch
        for (size_t i = 0; i < train_features.size(); i++) {
            // Copy features
            for (int j = 0; j < 4; j++) {
                batchFeatures[i][j] = train_features[i][0][j];
            }

            // One-hot encode labels
            for (int j = 0; j < 3; j++) {
                batchLabels[i][j] = (j == static_cast<int>(train_labels[i])) ? 1.0f : 0.0f;
            }
        }

        // Training step
        Tensor<float> inputGPU = batchFeatures.switchDevice(onGPU);
        Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));
        auto [loss, dOutput] = CategoricalCrossEntropyLoss(predictions, targetGPU);
        total_loss += loss;

        model.backward(dOutput);
        model.step();
        model.zeroGrad();

        // Evaluate on test set every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            float test_loss = 0.0f;
            float test_correct = 0.0f;

            // Prepare test batch
            Tensor<float> testFeatures(4, test_features.size(), false);
            Tensor<float> testLabels(3, test_features.size(), false);

            for (size_t i = 0; i < test_features.size(); i++) {
                for (int j = 0; j < 4; j++) {
                    testFeatures[i][j] = test_features[i][0][j];
                }
                for (int j = 0; j < 3; j++) {
                    testLabels[i][j] = (j == static_cast<int>(test_labels[i])) ? 1.0f : 0.0f;
                }
            }

            // Forward pass on test data
            Tensor<float> testInputGPU = testFeatures.switchDevice(onGPU);
            Tensor<float> testTargetGPU = testLabels.switchDevice(onGPU);

            Tensor<float> testPred = model.forward(std::move(testInputGPU));
            auto [test_loss_val, _] = CategoricalCrossEntropyLoss(testPred, testTargetGPU);

            // Compute accuracy
            Tensor<float> testPredCPU = testPred.switchDevice(false);
            for (size_t i = 0; i < test_features.size(); i++) {
                int predicted = 0;
                float max_val = testPredCPU[i][0];
                for (int j = 1; j < 3; j++) {
                    if (testPredCPU[i][j] > max_val) {
                        max_val = testPredCPU[i][j];
                        predicted = j;
                    }
                }
                if (predicted == static_cast<int>(test_labels[i])) test_correct += 1;
            }

            float test_accuracy = (test_correct / test_features.size()) * 100.0f;
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                      << ", Train Loss: " << total_loss
                      << ", Test Loss: " << test_loss_val
                      << ", Test Accuracy: " << test_accuracy << "%" << std::endl;
        }
    }
}

void train_linear_regression() {
    std::cout << "\n=== Training Linear Regression Model ===\n" << std::endl;

    // Load training data
    auto [train_features, train_labels] = TabularLoader::loadCSV(
        "/home/alex/CudaNeuralNetwork/onnx_generator/data/Regression/train.csv",  // Path to your training data
        ',',                 // delimiter
        true,                // normalize features
        -1,                  // load all rows
        true                 // skip header
    );

    // Load test data
    auto [test_features, test_labels] = TabularLoader::loadCSV(
        "/home/alex/CudaNeuralNetwork/onnx_generator/data/Regression/test.csv",   // Path to your test data
        ',',
        true,
        -1,
        true
    );

    // Create model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.01f, 0.0f)); // Learning rate 0.01, no momentum

    // Simple linear regression model: 1 input -> 1 output
    model.addLayer(std::make_unique<Linear>(1, 1, onGPU));

    // Training loop
    const int EPOCHS = 200;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Prepare training batch
        Tensor<float> batchFeatures(1, train_features.size(), false);
        Tensor<float> batchLabels(1, train_features.size(), false);

        for (size_t i = 0; i < train_features.size(); i++) {
            batchFeatures[i][0] = train_features[i][0][0];
            batchLabels[i][0] = train_labels[i];
        }

        // Training step
        Tensor<float> inputGPU = batchFeatures.switchDevice(onGPU);
        Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));
        auto [loss, dOutput] = computeMSELoss(predictions, targetGPU);

        model.backward(dOutput);
        model.step();
        model.zeroGrad();

        // Evaluate on test set every 20 epochs
        if ((epoch + 1) % 20 == 0) {
            // Prepare test batch
            Tensor<float> testFeatures(1, test_features.size(), false);
            Tensor<float> testLabels(1, test_features.size(), false);

            for (size_t i = 0; i < test_features.size(); i++) {
                testFeatures[i][0] = test_features[i][0][0];
                testLabels[i][0] = test_labels[i];
            }

            // Compute test loss
            Tensor<float> testInputGPU = testFeatures.switchDevice(onGPU);
            Tensor<float> testTargetGPU = testLabels.switchDevice(onGPU);

            Tensor<float> testPred = model.forward(std::move(testInputGPU));
            auto [test_loss, _] = computeMSELoss(testPred, testTargetGPU);

            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                      << ", Train Loss: " << loss
                      << ", Test Loss: " << test_loss << std::endl;

            // Print predictions for the last epoch
            if ((epoch + 1) == EPOCHS) {
                std::cout << "\nFinal test predictions:" << std::endl;
                Tensor<float> predCPU = testPred.switchDevice(false);
                for (size_t i = 0; i < test_features.size(); i++) {
                    std::cout << "Predicted: " << predCPU[i][0]
                              << ", Actual: " << test_labels[i] << std::endl;
                }
            }
        }
    }
}

int main() {
    Logger::setLevel(LogLevel::DEBUG);

    // Train both models
    train_iris();
    train_linear_regression();

    return 0;
}