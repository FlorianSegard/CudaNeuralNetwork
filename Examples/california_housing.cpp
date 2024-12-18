#include <iostream>
#include <random>

#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/MeanSquaredError/Mse.hpp"
#include "Loader/TabularLoader.hpp"
#include "Logger/Logger.hpp"

// Config
bool USE_GPU = false;
const int BATCH_SIZE = 3;
const int INPUT_FEATURES = 8;
const int HIDDEN_FEATURES = 64;
const int OUTPUT_FEATURES = 1;  // regression task
const int EPOCHS = 50;

// Helper function to prepare batch data for regression
std::pair<Tensor<float>, Tensor<float>> prepareBatch(
    const std::vector<Tensor<float>>& features,
    const std::vector<float>& targets,
    size_t startIdx,
    size_t batchSize) {

    Tensor<float> batchFeatures(INPUT_FEATURES, batchSize, false);
    Tensor<float> batchTargets(OUTPUT_FEATURES, batchSize, false);

    for (size_t b = 0; b < batchSize; b++) {
        // Copy feature data
        for (int x = 0; x < INPUT_FEATURES; x++) {
            batchFeatures[b][x] = features[startIdx + b][0][x];
        }

        // Set target value
        batchTargets[b][0] = targets[startIdx + b];
    }

    return {std::move(batchFeatures), std::move(batchTargets)};
}

// Helper function to compute MSE manually for evaluation
float computeMSE(const Tensor<float>& predictions, const std::vector<float>& targets, size_t startIdx, size_t batchSize) {
    float mse = 0.0f;

    for (size_t b = 0; b < batchSize; b++) {
        float diff = predictions[b][0] - targets[startIdx + b];
        mse += diff * diff;
    }

    return mse / batchSize;
}

std::vector<std::pair<float, float>> getSamplePredictions(
    Model& model,
    const std::vector<Tensor<float>>& features,
    const std::vector<float>& targets,
    int num_samples = 5) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, features.size() - 1);

    std::vector<std::pair<float, float>> predictions;

    for (int i = 0; i < std::min(num_samples, (int)features.size()); i++) {
        int idx = dis(gen);

        Tensor<float> input(INPUT_FEATURES, 1, false);
        for (int j = 0; j < INPUT_FEATURES; j++) {
            input[0][j] = features[idx][0][j];
        }

        Tensor<float> inputGPU = input.switchDevice(USE_GPU);
        Tensor<float> prediction = model.forward(std::move(inputGPU));
        Tensor<float> predCPU = prediction.switchDevice(false);

        predictions.push_back({targets[idx], predCPU[0][0]});
    }

    return predictions;
}

int main(int argc, char* argv[]) {
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
            USE_GPU = true;
        }
    }

    std::string train_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/California_Housing/train_data.csv";
    std::string test_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/California_Housing/test_data.csv";

    auto [train_features, train_targets] = TabularLoader::loadCSV(train_path, ',', true);
    auto [test_features, test_targets] = TabularLoader::loadCSV(test_path, ',', true);

    Model model;

    // momentum and L2 regularization
    model.setOptimizer(SGD(0.001f, 0.9f, 0.0001f));

    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, HIDDEN_FEATURES, USE_GPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(HIDDEN_FEATURES, HIDDEN_FEATURES, USE_GPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(HIDDEN_FEATURES, OUTPUT_FEATURES, USE_GPU));

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int batch_count = 0;

        for (size_t i = 0; i < train_features.size(); i += BATCH_SIZE) {
            size_t batchEnd = std::min(i + BATCH_SIZE, train_features.size());
            size_t batchSize = batchEnd - i;

            auto [batchFeatures, batchTargets] = prepareBatch(train_features, train_targets, i, batchSize);

            Tensor<float> inputGPU = batchFeatures.switchDevice(USE_GPU);
            Tensor<float> targetGPU = batchTargets.switchDevice(USE_GPU);

            // Forward pass
            Tensor<float> predictions = model.forward(std::move(inputGPU));

            auto [loss, dOutput] = computeMSELoss(predictions, targetGPU);
            total_loss += loss;
            batch_count++;

            // Backward pass and optimization
            model.backward(dOutput);
            model.step();
            model.zeroGrad();
        }

        // Print epoch statistics
        float avg_loss = total_loss / batch_count;
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Training Loss (MSE): " << avg_loss << std::endl;

        // Evaluate on test set every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            float test_loss = 0.0f;
            int test_batch_count = 0;

            for (size_t i = 0; i < test_features.size(); i += BATCH_SIZE) {
                size_t batchEnd = std::min(i + BATCH_SIZE, test_features.size());
                size_t batchSize = batchEnd - i;

                auto [batchFeatures, batchTargets] = prepareBatch(test_features, test_targets, i, batchSize);

                Tensor<float> inputGPU = batchFeatures.switchDevice(USE_GPU);
                Tensor<float> predictions = model.forward(std::move(inputGPU));

                Tensor<float> predCPU = predictions.switchDevice(false);
                test_loss += computeMSE(predCPU, test_targets, i, batchSize);
                test_batch_count++;
            }

            float avg_test_loss = test_loss / test_batch_count;
            std::cout << "Test Loss (MSE): " << avg_test_loss << std::endl;

            // visualization of sample predictions
            auto sample_predictions = getSamplePredictions(model, test_features, test_targets);
            std::cout << "\nSample Predictions vs Targets:" << std::endl;
            std::cout << std::fixed << std::setprecision(3);
            for (int i = 0; i < sample_predictions.size(); i++) {
                std::cout << "Sample " << i + 1
                          << " | Target: " << sample_predictions[i].first
                          << " | Prediction: " << sample_predictions[i].second
                          << " | Diff: " << std::abs(sample_predictions[i].first - sample_predictions[i].second)
                          << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return 0;
}