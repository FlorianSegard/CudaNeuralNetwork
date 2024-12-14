#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Softmax/Softmax.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/Loss.hpp"
#include "Loader/ModelLoader.hpp"
#include "MNIST/MNISTLoader.hpp"

const int BATCH_SIZE = 1;
const int INPUT_FEATURES = 784;  // 28x28 pixels
const int HIDDEN_FEATURES = 20;
const int OUTPUT_FEATURES = 10;  // 10 digits
const int TRAIN_SAMPLES = 10;
const int TEST_SAMPLES = 5;
const int EPOCHS = 10;

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

int main() {
    // Create model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.001f));

    // Add layers with ReLU activation
    model.addLayer(std::make_unique<Linear>(INPUT_FEATURES, HIDDEN_FEATURES, onGPU));
    model.addLayer(std::make_unique<ReLU>());
    model.addLayer(std::make_unique<Linear>(HIDDEN_FEATURES, OUTPUT_FEATURES, onGPU));
    model.addLayer(std::make_unique<Softmax>());

    // Load training data
    std::string train_images_path = "/home/alex/CudaNeuralNetwork/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string train_labels_path = "/home/alex/CudaNeuralNetwork/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    auto [train_images, train_labels] = MNISTLoader::loadMNIST(train_images_path, train_labels_path, false, TRAIN_SAMPLES);

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

            std::cout << "INPUT ";
            inputGPU.switchDevice(false).print();
            std::cout << "TARGET ";
            targetGPU.switchDevice(false).print();

            // Forward pass
            Tensor<float> predictions = model.forward(std::move(inputGPU));

            // Compute loss and gradient
            auto [loss, dOutput] = computeMSELoss(predictions, targetGPU);
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
        float avg_accuracy = (total_accuracy / batch_count) * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << ", Loss: " << avg_loss
                  << ", Training Accuracy: " << avg_accuracy << "%" << std::endl;
    }

    // Evaluation on test set
    std::string test_images_path = "/home/alex/CudaNeuralNetwork/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_path = "/home/alex/CudaNeuralNetwork/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    auto [test_images, test_labels] = MNISTLoader::loadMNIST(test_images_path, test_labels_path, false, TEST_SAMPLES);

    float test_accuracy = 0.0f;
    float test_loss = 0.0f;
    int test_batch_count = 0;

    for (size_t i = 0; i < test_images.size(); i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, test_images.size());
        size_t batchSize = batchEnd - i;

        auto [batchImages, batchLabels] = prepareBatch(test_images, test_labels, i, batchSize);

        Tensor<float> inputGPU = batchImages.switchDevice(onGPU);
        Tensor<float> targetGPU = batchLabels.switchDevice(onGPU);

        Tensor<float> predictions = model.forward(std::move(inputGPU));

        auto [loss, _] = computeMSELoss(predictions, targetGPU);
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