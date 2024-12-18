#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/Sigmoid/Sigmoid.hpp"
#include "Loss/BinaryCrossEntropy/BinCrossEntropy.hpp"
#include "Loader/TabularLoader.hpp"

// Config
const bool USE_GPU = true;
const int BATCH_SIZE = 1;
const int EPOCHS = 15;
const float LEARNING_RATE = 0.01f;
const float WEIGHT_DECAY = 0.01f;

// Helper function to prepare batch data
std::pair<Tensor<float>, Tensor<float>> prepareBatch(
    const std::vector<Tensor<float>>& features,
    const std::vector<float>& labels,
    size_t startIdx,
    size_t batchSize) {

    int numFeatures = features[0].width;
    Tensor<float> batchFeatures(numFeatures, batchSize, false);
    Tensor<float> batchLabels(1, batchSize, false);

    for (size_t b = 0; b < batchSize; b++) {
        // Copy feature data
        for (int x = 0; x < numFeatures; x++) {
            batchFeatures[b][x] = features[startIdx + b][0][x];
        }
        // Set label
        batchLabels[b][0] = labels[startIdx + b];
    }

    return {std::move(batchFeatures), std::move(batchLabels)};
}

// Helper function to compute accuracy
float computeAccuracy(const Tensor<float>& predictions, const std::vector<float>& labels, size_t startIdx, size_t batchSize) {
    int correct = 0;

    for (size_t b = 0; b < batchSize; b++) {
        float predicted = predictions[b][0] >= 0.5f ? 1.0f : 0.0f;
        if (predicted == labels[startIdx + b]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batchSize;
}

int main(int argc, char* argv[]) {
    std::string train_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/Breast_Cancer/train_data.csv";
    std::string test_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/Breast_Cancer/test_data.csv";

    auto [train_features, train_labels] = TabularLoader::loadCSV(train_path, ',', true);
    auto [test_features, test_labels] = TabularLoader::loadCSV(test_path, ',', true);

    const int INPUT_FEATURES = train_features[0].width;

    // Logistic regression
    Model model;
    model.setOptimizer(SGD(LEARNING_RATE, 0.9f, WEIGHT_DECAY)); // Using momentum and L2 regularization

    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, 1, USE_GPU));
    model.addLayer(std::make_unique<Sigmoid>(true)); // true indicates it's used with BCE loss

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        float total_accuracy = 0.0f;
        int batch_count = 0;

        for (size_t i = 0; i < train_features.size(); i += BATCH_SIZE) {
            size_t batchEnd = std::min(i + BATCH_SIZE, train_features.size());
            size_t batchSize = batchEnd - i;

            auto [batchFeatures, batchLabels] = prepareBatch(train_features, train_labels, i, batchSize);

            Tensor<float> inputGPU = batchFeatures.switchDevice(USE_GPU);
            Tensor<float> targetGPU = batchLabels.switchDevice(USE_GPU);

            // Forward pass
            Tensor<float> predictions = model.forward(std::move(inputGPU));

            // Compute loss and gradient
            auto [loss, dOutput] = BinaryCrossEntropyLoss(predictions, targetGPU);
            total_loss += loss;

            Tensor<float> predCPU = predictions.switchDevice(false);
            total_accuracy += computeAccuracy(predCPU, train_labels, i, batchSize);
            batch_count++;

            // Backward pass and optimization
            model.backward(dOutput);
            model.step();
            model.zeroGrad();
        }

        float avg_loss = total_loss / batch_count;
        float avg_accuracy = (total_accuracy / batch_count) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Loss: " << avg_loss
                  << ", Training Accuracy: " << avg_accuracy << "%" << std::endl;
    }

    // Evaluation on test set
    float test_accuracy = 0.0f;
    float test_loss = 0.0f;
    int test_batch_count = 0;

    for (size_t i = 0; i < test_features.size(); i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, test_features.size());
        size_t batchSize = batchEnd - i;

        auto [batchFeatures, batchLabels] = prepareBatch(test_features, test_labels, i, batchSize);

        Tensor<float> inputGPU = batchFeatures.switchDevice(USE_GPU);
        Tensor<float> targetGPU = batchLabels.switchDevice(USE_GPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));

        auto [loss, _] = BinaryCrossEntropyLoss(predictions, targetGPU);
        test_loss += loss;

        Tensor<float> predCPU = predictions.switchDevice(false);
        test_accuracy += computeAccuracy(predCPU, test_labels, i, batchSize);
        test_batch_count++;
    }

    float final_test_loss = test_loss / test_batch_count;
    float final_test_accuracy = (test_accuracy / test_batch_count) * 100.0f;

    std::cout << "\nTest Results:" << std::endl;
    std::cout << "Loss: " << final_test_loss << std::endl;
    std::cout << "Accuracy: " << final_test_accuracy << "%" << std::endl;

    return 0;
}