#pragma once
#include "ONNXLoader.hpp"
#include <string>
#include <fstream>
#include <iostream>

#include "Model/Model.hpp"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

class ONNXLoader {
private:
    static void analyzeGraph(const onnx::GraphProto& graph) {
        Logger::debug("\n====== ONNX Graph Analysis ======");
        Logger::debug("Inputs:");
        for (const auto& input : graph.input()) {
            Logger::debug(" - " + input.name());
        }

        Logger::debug("\nOutputs:");
        for (const auto& output : graph.output()) {
            Logger::debug(" - " + output.name());
        }

        Logger::debug("\nNodes:");
        for (const auto& node : graph.node()) {
            Logger::debug(" - " + node.op_type() + " (" + node.name() + ")");
        }
    }

public:
    static void onnxPrettyPrint(const std::string& model_path) {
        onnx::ModelProto model_proto;
        std::ifstream in(model_path, std::ios_base::binary);

        if (!in.is_open()) {
            throw std::runtime_error("Failed to open model file: " + model_path);
        }

        if (!model_proto.ParseFromIstream(&in)) {
            throw std::runtime_error("Failed to parse ONNX model");
        }
        in.close();

        std::cout << "\n======= loadONNX model =======" << "\n";

        const auto& graph = model_proto.graph();

        analyzeGraph(graph);
    }

    static Model loadONNX(const std::string& model_path, bool useGPU = false) {
        onnx::ModelProto model_proto;
        std::ifstream in(model_path, std::ios_base::binary);

        if (!in.is_open()) {
            throw std::runtime_error("Failed to open model file: " + model_path);
        }

        if (!model_proto.ParseFromIstream(&in)) {
            throw std::runtime_error("Failed to parse ONNX model");
        }
        in.close();

        std::cout << "\n======= loadONNX model =======" << "\n";

        const auto& graph = model_proto.graph();
        std::vector<std::unique_ptr<Layer>> layers;

        // Map to store initialized tensors (weights and biases)
        std::unordered_map<std::string, const onnx::TensorProto*> initializers;
        for (const auto& initializer : graph.initializer()) {
            initializers[initializer.name()] = &initializer;
        }

        for (const auto& node : graph.node()) {
            if (node.op_type() == "Gemm" || node.op_type() == "Linear") {
                const auto* weight_tensor = initializers[node.input(1)];
                const auto* bias_tensor = initializers[node.input(2)];

                long input_size = weight_tensor->dims(1);
                long output_size = weight_tensor->dims(0);

                std::cout << "Layer: " << node.name() << " -> Input size: " << input_size;
                std::cout << ", Output size: " << output_size << "\n";

                auto layer = std::make_unique<Linear>(input_size, output_size, useGPU);

                const auto* weight_data = reinterpret_cast<const float*>(weight_tensor->raw_data().data());
                for (long i = 0; i < input_size; ++i) {
                    for (long j = 0; j < output_size; ++j) {
                        layer->params.weights[i][j] = weight_data[input_size * i + j];
                    }
                }
                std::cout << "Weight: loaded";

                const auto* bias_data = reinterpret_cast<const float*>(bias_tensor->raw_data().data());
                for (long i = 0; i < output_size; ++i) {
                    layer->params.biases[0][i] = bias_data[i];
                }
                std::cout << ", Biases: loaded" << "\n";

                layers.push_back(std::move(layer));
            }
        }

        return Model(std::move(layers));
    }
};