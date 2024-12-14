#pragma once
#include "../Layers.hpp"
#include <iostream>
#include <typeinfo>


struct Linear : public Layer {


    Linear(int inputSize, int outputSize, bool device = false, bool require_grad = true)
            : Layer(inputSize, outputSize, device, require_grad) {
                this->params.weights.fillOnes();
                this->params.weights = this->params.weights * 0.5;
                this->params.biases.fillZero();
            }

    Tensor<float> computeForward(Tensor<float>& input) override {
        // Linear forward: output = input @ weights.T + biases
        //     std::cout << "------------------------- ENTERING FORWARD -------------------------" << std::endl;

        //     std::cout << "5" << std::endl;

        //     std::cout << "width params.WEIGHTS tensor" << " " << params.weights.width << std::endl;
        //     std::cout << "height params.WEIGHTS tensor" << " " << params.weights.height << std::endl;

        //     std::cout << "width INPUT tensor" << " " << input.width << std::endl;
        //     std::cout << "height INPUT tensor" << " " << input.height << std::endl;

        // Dot matrix product [batch_size, input_size] @ [input_size, output_size]
        Tensor<float> output = input.dot(params.weights);
            // std::cout << "6" << std::endl;

            // std::cout << "width OUTPUT tensor" << " " << output.width << std::endl;
            // std::cout << "height OUTPUT tensor" << " " << output.height << std::endl;

            // std::cout << "width params.BIASES tensor" << " " << params.biases.width << std::endl;
            // std::cout << "height params.BIASES tensor" << " " << params.biases.height << std::endl;
        // Add biases
        return output + params.biases;
    }

    Tensor<float> backward(Tensor<float>& dOutput) override {
        // dInput = dOutput @ weights
        if (!require_grad) return Tensor<float>();

            // std::cout << "------------------------- ENTERING BACWARD -------------------------" << std::endl;
            // std::cout << "width dOutput tensor" << " " << dOutput.width << std::endl;
            // std::cout << "height dOutput tensor" << " " << dOutput.height << std::endl;

            // std::cout << "width params.WEIGHTS tensor" << " " << params.weights.width << std::endl;
            // std::cout << "height params.WEIGHTS tensor" << " " << params.weights.height << std::endl;
        Tensor<float> dInput = dOutput.dot(params.weights.transpose());

        // dWeights = input.T @ dOutput
        Tensor<float> inputT = this->lastInput.transpose();
        //     std::cout << "width inputT tensor" << " " << inputT.width << std::endl;
        //     std::cout << "height inputT tensor" << " " << inputT.height << std::endl;

        //     std::cout << "width params.dWeights tensor" << " " << params.dWeights.width << std::endl;
        //     std::cout << "height params.dWeights tensor" << " " << params.dWeights.height << std::endl;

        params.dWeights = inputT.dot(dOutput);
        //     // params.dBiases = dOutput.clone();

        //     std::cout << "width params.dWeights tensor" << " " << params.dWeights.width << std::endl;
        //     std::cout << "height params.dWeights tensor" << " " << params.dWeights.height << std::endl;



        //     std::cout << "width params.dBiases tensor" << " " << params.dBiases.width << std::endl;
        //     std::cout << "height params.dBiases tensor" << " " << params.dBiases.height << std::endl;
        // Calculate input gradients for backprop
        return dOutput.dot(params.weights.transpose());
    }
};