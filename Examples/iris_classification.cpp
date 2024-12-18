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

int main(int argc, char* argv[]) {
    bool onGPU = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--infer") {
            Logger::setLevel(LogLevel::INFER);
        } else if (arg == "-b" || arg == "--back") {
            Logger::setLevel(LogLevel::BACKPROP);
        } else if (arg == "-l" || arg == "--loss") {
            Logger::setLevel(LogLevel::LOSS);
        } else if (arg == "-d" || arg == "--debug") {
            Logger::setLevel(LogLevel::DEBUG);
        } else if (arg == "-a" || arg == "--all") {
            Logger::setLevel(LogLevel::ALL);
        } else if (arg == "--gpu") {
            std::cout << "-- Using GPU --" << std::endl;
            onGPU = true;
        }
    }
    std::cout << "\n=== Training Iris Classification Model ===\n" << std::endl;

    std::string train_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/Iris/train_data.csv";
    std::string test_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/Iris/test_data.csv";


    auto [train_features, train_labels] = TabularLoader::loadCSV(train_path, ',', false);
    auto [test_features, test_labels] = TabularLoader::loadCSV(test_path, ',', false);

    Model model;

    // Learning rate 0.01, momentum 0.9
    model.setOptimizer(SGD(0.01f, 0.9));

    // Architecture for Iris classification
    // 4 input features -> 8 hidden -> 3 classes
    model.addLayer(std::make_unique<Linear>(4, 8, onGPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(8, 3, onGPU));
    model.addLayer(std::make_unique<Softmax>(true));

    const int EPOCHS = 500;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        float correct = 0.0f;

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
                      << ", train_loss: " << total_loss
                      << ", test_loss: " << test_loss_val
                      << ", test_accuracy: " << test_accuracy << "%" << std::endl;
        }
    }

    return 0;
}