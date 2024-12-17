#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Softmax/Softmax.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/CategoricalCrossEntropy/CatCrossEntropy.hpp"
// #include "Loader/ModelLoader.hpp"
#include "Logger/Logger.hpp"
#include "MNIST/MNISTLoader.hpp"
#include "Scheduler/Scheduler.hpp"
#include "Layers/Dropout/Dropout.hpp"

const int BATCH_SIZE = 32;
const int INPUT_FEATURES = 784;  // 28x28 pixels
const int HIDDEN_FEATURES = 20;
const int HIDDEN_FEATURES_2 = 20;
const int OUTPUT_FEATURES = 10;  // 10 digits
const int TRAIN_SAMPLES = 10000;
const int TEST_SAMPLES = 1000;
const int EPOCHS = 30;

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
        }
    }

    // Create model
    bool onGPU = false;
    Model model;
    SGD optimizer = SGD(0.01f, 0.0f);

    model.setOptimizer(optimizer);

    ReduceLROnPlateau scheduler = ReduceLROnPlateau(&optimizer);

    // Add layers with ReLU activation
    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, HIDDEN_FEATURES, onGPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Dropout>(0.5f));
    model.addLayer(std::make_unique<Linear>(HIDDEN_FEATURES, OUTPUT_FEATURES, onGPU));
    model.addLayer(std::make_unique<Softmax>(true));

    // Load training data
    std::string train_images_path = "/home/florian/CudaNeuralNetwork/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string train_labels_path = "/home/florian/CudaNeuralNetwork/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    auto [train_images, train_labels] = MNISTLoader::loadMNIST(train_images_path, train_labels_path, true, TRAIN_SAMPLES);


    model.trainMode();
    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        float total_accuracy = 0.0f;
        int batch_count = 0;

        for (size_t i = 0; i < train_images.size(); i += BATCH_SIZE) {
            size_t batchEnd = std::min(i + BATCH_SIZE, train_images.size());
            size_t batchSize = batchEnd - i;

            // Prepare batch
            auto [batchImages, batchLabels] = prepareBatch(train_images, train_labels, i, batchSize);

            // Move to GPU if needed
            Tensor<float> inputGPU = batchImages.switchDevice(onGPU);
            Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);


            // Forward pass
            Tensor<float> predictions = model.forward(std::move(inputGPU));

            // Compute loss and gradient
            auto [loss, dOutput] = CategoricalCrossEntropyLoss(predictions, targetGPU);
            total_loss += loss;

            // Compute accuracy
            Tensor<float> predCPU = predictions.switchDevice(false);
            total_accuracy += computeAccuracy(predCPU, train_labels, i, batchSize);
            batch_count++;

            // Backward pass and optimization
            model.backward(dOutput);
            model.step();


            //check_weights(&model);
            model.zeroGrad();
        }

        // Print epoch statistics
        float avg_loss = total_loss / batch_count;
        scheduler.step(avg_loss);

        float avg_accuracy = (total_accuracy / batch_count) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Loss: " << avg_loss
                  << ", Training Accuracy: " << avg_accuracy << "%" << std::endl;
    }
    freeCurandStates();

    // Evaluation on test set
    std::string test_images_path = "/home/florian/CudaNeuralNetwork/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_path = "/home/florian/CudaNeuralNetwork/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    auto [test_images, test_labels] = MNISTLoader::loadMNIST(test_images_path, test_labels_path, true, TEST_SAMPLES);

    float test_accuracy = 0.0f;
    float test_loss = 0.0f;
    int test_batch_count = 0;
    std::cout << "eval mode" << std::endl;

    model.evalMode();
    for (size_t i = 0; i < test_images.size(); i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, test_images.size());
        size_t batchSize = batchEnd - i;

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

    std::cout << "Test Results:" << std::endl;
    std::cout << "Loss: " << final_test_loss << std::endl;
    std::cout << "Accuracy: " << final_test_accuracy << "%" << std::endl;
    return 0;
}