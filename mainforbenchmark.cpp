#include <iostream>
#include <fstream>
#include <chrono> // Added for benchmarking time
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Softmax/Softmax.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/CategoricalCrossEntropy/CatCrossEntropy.hpp"
#include "Logger/Logger.hpp"
#include "MNIST/MNISTLoader.hpp"
#include "Scheduler/Scheduler.hpp"

const int BATCH_SIZE = 32;
const int INPUT_FEATURES = 784;
const int HIDDEN_FEATURES = 20;
const int OUTPUT_FEATURES = 10;
const int TRAIN_SAMPLES = 10000;
const int TEST_SAMPLES = 1000;
const int EPOCHS = 30;







// Helper function to prepare batch data
std::pair<Tensor<float>, Tensor<float>> prepareBatch(
    const std::vector<Tensor<float>>& images,
    const std::vector<int>& labels,
    size_t startIdx,
    size_t batchSize) {

    Tensor<float> batchImages(INPUT_FEATURES, batchSize, false);
    Tensor<float> batchLabels(OUTPUT_FEATURES, batchSize, false);

    for (size_t b = 0; b < batchSize; b++) {
        // Copy image data
        for (int x = 0; x < INPUT_FEATURES; x++) {
            batchImages[b][x] = images[startIdx + b][0][x];
        }

        // Create one-hot encoded labels
        for (int x = 0; x < OUTPUT_FEATURES; x++) {
            batchLabels[b][x] = (x == labels[startIdx + b]) ? 1.0f : 0.0f;
        }
    }

    return {std::move(batchImages), std::move(batchLabels)};
}

// Helper function to compute accuracy
float computeAccuracy(const Tensor<float>& predictions, const std::vector<int>& labels, size_t startIdx, size_t batchSize) {
    int correct = 0;

    for (size_t b = 0; b < batchSize; b++) {
        int predicted_digit = 0;
        float max_val = predictions[b][0];

        for (int k = 1; k < OUTPUT_FEATURES; k++) {
            if (predictions[b][k] > max_val) {
                max_val = predictions[b][k];
                predicted_digit = k;
            }
        }

        if (predicted_digit == labels[startIdx + b]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batchSize;
}

void logMetrics(std::ofstream& logFile, int epoch, float loss, float accuracy) {
    logFile << "Epoch: " << epoch
            << ", Loss: " << loss
            << ", Accuracy: " << accuracy << "%" << std::endl;
}

void logFinalTestMetrics(std::ofstream& logFile, float loss, float accuracy) {
    logFile << "Final Test Results:" << std::endl;
    logFile << "Loss: " << loss << std::endl;
    logFile << "Accuracy: " << accuracy << "%" << std::endl;
}

int main() {
    std::ofstream logFile("training_log.txt"); // Create and open the log file
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file!" << std::endl;
        return -1;
    }

    bool onGPU = false;
    Model model;
    SGD optimizer = SGD(0.01f, 0.0f);
    model.setOptimizer(optimizer);
    ReduceLROnPlateau scheduler = ReduceLROnPlateau(&optimizer);

    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, HIDDEN_FEATURES, onGPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(HIDDEN_FEATURES, OUTPUT_FEATURES, onGPU));
    model.addLayer(std::make_unique<Softmax>(true));

    auto [train_images, train_labels] = MNISTLoader::loadMNIST(
        "/home/florian/CudaNeuralNetwork/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte", 
        "/home/florian/CudaNeuralNetwork/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte", true, TRAIN_SAMPLES);

    auto start_time = std::chrono::high_resolution_clock::now(); // Start training timer

    model.trainMode();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f, total_accuracy = 0.0f;
        int batch_count = 0;

        for (size_t i = 0; i < train_images.size(); i += BATCH_SIZE) {
            size_t batchSize = std::min(BATCH_SIZE, static_cast<int>(train_images.size() - i));
            auto [batchImages, batchLabels] = prepareBatch(train_images, train_labels, i, batchSize);

            Tensor<float> inputGPU = batchImages.switchDevice(onGPU);
            Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);

            Tensor<float> predictions = model.forward(std::move(inputGPU));
            auto [loss, dOutput] = CategoricalCrossEntropyLoss(predictions, targetGPU);

            total_loss += loss;
            Tensor<float> predCPU = predictions.switchDevice(false);
            total_accuracy += computeAccuracy(predCPU, train_labels, i, batchSize);

            model.backward(dOutput);
            model.step();
            model.zeroGrad();

            batch_count++;
        }

        float avg_loss = total_loss / batch_count;
        float avg_accuracy = (total_accuracy / batch_count) * 100.0f;
        scheduler.step(avg_loss);

        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Loss: " << avg_loss
                  << ", Accuracy: " << avg_accuracy << "%" << std::endl;

        logMetrics(logFile, epoch + 1, avg_loss, avg_accuracy); // Log to file
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // End training timer
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Total Training Time: " << duration.count() << " seconds" << std::endl;
    logFile << "Total Training Time: " << duration.count() << " seconds" << std::endl;

    auto [test_images, test_labels] = MNISTLoader::loadMNIST(
        "/home/florian/CudaNeuralNetwork/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 
        "/home/florian/CudaNeuralNetwork/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte", true, TEST_SAMPLES);

    float test_loss = 0.0f, test_accuracy = 0.0f;
    int test_batch_count = 0;

    model.evalMode();
    for (size_t i = 0; i < test_images.size(); i += BATCH_SIZE) {
        size_t batchSize = std::min(BATCH_SIZE, static_cast<int>(test_images.size() - i));
        auto [batchImages, batchLabels] = prepareBatch(test_images, test_labels, i, batchSize);

        Tensor<float> inputGPU = batchImages.switchDevice(onGPU);
        Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));
        auto [loss, _] = CategoricalCrossEntropyLoss(predictions, targetGPU);

        test_loss += loss;
        Tensor<float> predCPU = predictions.switchDevice(false);
        test_accuracy += computeAccuracy(predCPU, test_labels, i, batchSize);
        test_batch_count++;
    }

    float final_test_loss = test_loss / test_batch_count;
    float final_test_accuracy = (test_accuracy / test_batch_count) * 100.0f;

    std::cout << "Test Loss: " << final_test_loss << ", Test Accuracy: " << final_test_accuracy << "%" << std::endl;
    logFinalTestMetrics(logFile, final_test_loss, final_test_accuracy); // Log final test results

    logFile.close();
    return 0;
}
