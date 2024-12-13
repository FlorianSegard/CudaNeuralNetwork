#include <iostream>
#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Tensor/Tensor.hpp"
#include "Loss/Loss.hpp"
#include "Loader/ModelLoader.hpp"

int main() {
    // Create a model
    bool onGPU = true;
    Model model;
    model.setOptimizer(SGD(0.01f));
    std::cout << "1.1" << std::endl;

    // Suppose our input is size (batch_size=2, input_features=3) and we want an output of size (2, 2).
    // We'll chain two Linear layers: Linear(3->4) then Linear(4->2)
    // We'll keep them on CPU for simplicity: device = false
    model.addLayer(std::make_unique<Linear>(3, 4, onGPU));
    model.addLayer(std::make_unique<Linear>(4, 2, onGPU));
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

        Tensor<float> inputGPU = input.switchDevice(onGPU);
        Tensor<float> targetGPU = target.switchDevice(onGPU);

        


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
            model.switchDevice(false);

            Linear* linear1 = dynamic_cast<Linear*>(model.layers[0].get());
            if (linear1) {
                std::cout << "First layer weights after step:" << std::endl;
                for (int y = 0; y < linear1->params.weights.height; y++) {
                    for (int x = 0; x < linear1->params.weights.width; x++) {
                        std::cout << linear1->params.weights[y][x] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            model.switchDevice(true);

        }

    }

    Model model_onnx = ModelLoader::loadONNX("/home/alex/CudaNeuralNetwork/onnx_generator/simple_linear_model.onnx", true);
    // Print out something from model parameters to verify changes
    // For instance, if you have Linear layers, you can inspect their params
    // Assuming the first layer is a Linear layer:
    // model.layers[0] is a std::unique_ptr<Layer>, dynamic_cast to Linear:
    std::cout << "=== END ==="<<std::endl;
    return 0;
}
