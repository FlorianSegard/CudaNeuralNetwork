#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/Loss.hpp"
#include "Loader/ModelLoader.hpp"
#include "MNIST/MNISTLoader.hpp"

int main() {
    // Create a model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.01f));

    // MNIST input is 784 (28x28 flattened), output is 10 (digits 0-9)
    model.addLayer(std::make_unique<Linear>(784, 100, onGPU));  // First layer: 784->100
    model.addLayer(std::make_unique<Linear>(100, 10, onGPU));   // Second layer: 100->10

    // Load training data
    std::string train_images_path = "/home/alex/CudaNeuralNetwork/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string train_labels_path = "/home/alex/CudaNeuralNetwork/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    auto [train_images, train_labels] = MNISTLoader::loadMNIST(train_images_path, train_labels_path, true, 1000);  // Load 1000 training examples

    // Training loop
    int epochs = 10;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < train_images.size(); i++) {
            // Prepare input
            Tensor<float> input_gpu = train_images[i].switchDevice(onGPU);

            // Prepare target (one-hot encoded)
            Tensor<float> target = MNISTLoader::labelToOneHot(train_labels[i]);
            Tensor<float> target_gpu = target.switchDevice(onGPU);

            // Forward pass
            Tensor<float> predictions = model.forward(std::move(input_gpu));

            // Compute loss and gradient
            auto [loss, dOutput] = computeMSELoss(predictions, target_gpu);
            total_loss += loss;

            // Backward pass and update
            model.backward(dOutput);
            model.step();

            // Track accuracy (move predictions to CPU for evaluation)
            Tensor<float> pred_cpu = predictions.switchDevice(false);
            int predicted_digit = 0;
            float max_val = pred_cpu[0][0];
            for (int k = 1; k < 10; k++) {
                if (pred_cpu[0][k] > max_val) {
                    max_val = pred_cpu[0][k];
                    predicted_digit = k;
                }
            }
            if (predicted_digit == train_labels[i]) {
                correct++;
            }
        }

        // Print epoch statistics
        float avg_loss = total_loss / train_images.size();
        float accuracy = (float)correct / train_images.size() * 100.0f;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << avg_loss
                  << ", Training Accuracy: " << accuracy << "%" << std::endl;
    }

    // Evaluation on test set
    std::string test_images_path = "/home/alex/CudaNeuralNetwork/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_path = "/home/alex/CudaNeuralNetwork/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    auto [test_images, test_labels] = MNISTLoader::loadMNIST(test_images_path, test_labels_path, true, 100);

    model.switchDevice(onGPU);
    int num_correct = 0;
    int total = 0;

    for (size_t i = 0; i < test_images.size(); i++) {
        Tensor<float> image_gpu = test_images[i].switchDevice(true);
        Tensor<float> prediction = model.forward(std::move(image_gpu));
        prediction = prediction.switchDevice(false);

        int predicted_digit = 0;
        float max_val = prediction[0][0];
        for (int k = 1; k < 10; k++) {
            if (prediction[0][k] > max_val) {
                max_val = prediction[0][k];
                predicted_digit = k;
            }
        }

        if (predicted_digit == test_labels[i]) {
            num_correct++;
        }
        total++;
    }

    float accuracy = (float)num_correct / total * 100.0f;
    std::cout << "Test Accuracy: " << accuracy << "% (" << num_correct << "/" << total << ")" << std::endl;

    return 0;
}