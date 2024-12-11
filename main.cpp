#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Tensor/Tensor.hpp"

// A dummy Mean Squared Error (MSE) loss function
// Returns the loss value and computes dLoss/dOutput for backward.
std::pair<float, Tensor<float>> computeMSELoss(Tensor<float>& predictions, Tensor<float>& targets) {
    // Ensure dimensions match
    if (predictions.width != targets.width || predictions.height != targets.height) {
        throw std::invalid_argument("Dimensions do not match for loss calculation.");
    }

    // We'll assume CPU for simplicity. If it's on GPU, you'd transfer to CPU or implement a GPU kernel.
    // predictions and targets should be on the same device for simplicity.
    bool onGPU = predictions.device;
    Tensor<float> predCPU = onGPU ? predictions.switchDevice(false) : predictions.clone();
    Tensor<float> targetsCPU = onGPU ? targets.switchDevice(false) : targets.clone();

    float loss = 0.0f;
    // Compute MSE: loss = sum((pred - target)^2) / (N)
    for (int y = 0; y < predCPU.height; ++y) {
        for (int x = 0; x < predCPU.width; ++x) {
            float diff = predCPU[y][x] - targetsCPU[y][x];
            loss += diff * diff;
        }
    }
    loss /= (predCPU.width * predCPU.height);

    // Compute gradient dLoss/dPred = 2*(pred - target)/N
    Tensor<float> grad(predictions.width, predictions.height, onGPU);
    if (onGPU) {
        // If on GPU, you would copy data back and forth or implement a kernel.
        // For simplicity, let's just compute on CPU and then move the gradient to GPU.
        for (int y = 0; y < predCPU.height; ++y) {
            for (int x = 0; x < predCPU.width; ++x) {
                float diff = predCPU[y][x] - targetsCPU[y][x];
                diff *= (2.0f / (predCPU.width * predCPU.height));
                grad[y][x] = diff;
            }
        }
        grad = grad.switchDevice(true);
    } else {
        for (int y = 0; y < predCPU.height; ++y) {
            for (int x = 0; x < predCPU.width; ++x) {
                float diff = predCPU[y][x] - targetsCPU[y][x];
                diff *= (2.0f / (predCPU.width * predCPU.height));
                grad[y][x] = diff;
            }
        }
    }

    return {loss, std::move(grad)};
}

int main() {
    // Create a model
    Model model;
    model.setOptimizer(SGD(0.01f));
    std::cout << "1.1" << std::endl;

    // Suppose our input is size (batch_size=2, input_features=3) and we want an output of size (2, 2).
    // We'll chain two Linear layers: Linear(3->4) then Linear(4->2)
    // We'll keep them on CPU for simplicity: device = false
    model.addLayer(std::make_unique<Linear>(3, 4, false));
    model.addLayer(std::make_unique<Linear>(4, 2, false));
    std::cout << "1.2" << std::endl;


    // Forward pass
    for (int k = 0; k < 10; k++)
    {

        // Create a small batch of inputs (2 examples, 3 features each)
        Tensor<float> input(3, 2, false); // width=3, height=2
        std::cout << "1.3" << std::endl;

        // Fill input with some values:
        // input is indexed as input[y][x], y in [0,1], x in [0,2]
        // We'll treat "height" as batch dimension and "width" as feature dimension for demonstration
        // e.g. input for sample 0: input[0][0..2]
        //      input for sample 1: input[1][0..2]
        input[0][0] = 0.5f; input[0][1] = -1.0f; input[0][2] = 2.0f;
        input[1][0] = 1.0f; input[1][1] = 0.5f; input[1][2] = -0.5f;
        std::cout << "1.4" << std::endl;

        // Create a target for this batch (2 examples, 2 features)
        Tensor<float> target(2, 2, false); // width=2, height=2
        // target[y][x]
        target[0][0] = 0.0f; target[0][1] = 1.0f;
        target[1][0] = 1.0f; target[1][1] = 0.0f;
        std::cout << "2" << std::endl;

        Tensor<float> inputGPU = input.switchDevice(false);
        Tensor<float> targetGPU = target.switchDevice(false);

        


        Tensor<float> predictions = model.forward(std::move(inputGPU));
        // Tensor<float> predictions = model.forward(std::move(input));
        std::cout << "3" << std::endl;

        // Compute loss and gradient w.r.t. predictions
        auto [loss, dOutput] = computeMSELoss(predictions, targetGPU);

        std::cout << "Loss: " << loss << std::endl;

        // Backward pass
        model.backward(dOutput);
        std::cout << "end backward: " << std::endl;

        // Update step
        model.step();
        std::cout << "model step: " << std::endl;
        {
            Linear* linear1 = dynamic_cast<Linear*>(model.layers[0].get());
            if (linear1) {
                std::cout << "First layer weights after step:" << std::endl;
                for (int y = 0; y < linear1->params.weights.height; ++y) {
                    for (int x = 0; x < linear1->params.weights.width; ++x) {
                        std::cout << linear1->params.weights[y][x] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

    }

    // Print out something from model parameters to verify changes
    // For instance, if you have Linear layers, you can inspect their params
    // Assuming the first layer is a Linear layer:
    // model.layers[0] is a std::unique_ptr<Layer>, dynamic_cast to Linear:

    return 0;
}
