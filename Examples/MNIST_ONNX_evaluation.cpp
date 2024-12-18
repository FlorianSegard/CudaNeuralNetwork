#include <iostream>
#include <vector>
#include <chrono>
#include "Model/Model.hpp"
#include "Loader/MNISTLoader.hpp"
#include "Loader/ONNXLoader.hpp"

void evaluateModel(Model& model, const std::vector<Tensor<float>>& images,
                  const std::vector<int>& labels, bool useGPU,
                  const std::string& modelName) {

    const int BATCH_SIZE = 32;
    float total_accuracy = 0.0f;
    int batch_count = 0;

    model.evalMode();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); i += BATCH_SIZE) {
        size_t batchEnd = std::min(i + BATCH_SIZE, images.size());
        size_t batchSize = batchEnd - i;

        Tensor<float> batchInput(784, batchSize, false);
        for (size_t b = 0; b < batchSize; b++) {
            for (int x = 0; x < 784; x++) {
                batchInput[b][x] = images[i + b][0][x];
            }
        }

        if (useGPU) {
            batchInput = batchInput.switchDevice(true);
        }

        Tensor<float> predictions = model.forward(std::move(batchInput));

        if (useGPU) {
            predictions = predictions.switchDevice(false);
        }

        int correct = 0;
        for (size_t b = 0; b < batchSize; b++) {
            int predicted_digit = 0;
            float max_val = predictions[b][0];

            for (int k = 1; k < 10; k++) {
                if (predictions[b][k] > max_val) {
                    max_val = predictions[b][k];
                    predicted_digit = k;
                }
            }

            if (predicted_digit == labels[i + b]) {
                correct++;
            }
        }

        total_accuracy += static_cast<float>(correct) / batchSize;
        batch_count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    float final_accuracy = (total_accuracy / batch_count) * 100.0f;
    std::cout << "\n" << modelName << " Results:" << std::endl;
    std::cout << "Accuracy: " << final_accuracy << "%" << std::endl;
    std::cout << "Evaluation time: " << duration.count() << "ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    bool useGPU = false;

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
            useGPU = true;
        }
    }

    // Load test dataset
    std::string test_images_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/MNIST/raw/t10k-images-idx3-ubyte";
    std::string test_labels_path = "/home/alex/CudaNeuralNetwork/onnx_generator/data/MNIST/raw/t10k-labels-idx1-ubyte";

    auto [test_images, test_labels] = MNISTLoader::loadMNIST(
        test_images_path, test_labels_path, true, 10000
    );
    std::cout << "Loaded " << test_images.size() << " test images" << std::endl;

    try {
        ONNXLoader loader;

        // Load and evaluate untrained model
        std::cout << "\nLoading untrained model..." << std::endl;
        loader.onnxPrettyPrint("/home/alex/CudaNeuralNetwork/onnx_generator/mnist_untrained.onnx");
        Model untrained_model = loader.loadONNX("/home/alex/CudaNeuralNetwork/onnx_generator/mnist_untrained.onnx", useGPU);
        evaluateModel(untrained_model, test_images, test_labels, useGPU, "Untrained Model");

        // Load and evaluate trained model
        std::cout << "\nLoading trained model..." << std::endl;
        Model trained_model = loader.loadONNX("/home/alex/CudaNeuralNetwork/onnx_generator/mnist_trained.onnx", useGPU);
        evaluateModel(trained_model, test_images, test_labels, useGPU, "Trained Model");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}