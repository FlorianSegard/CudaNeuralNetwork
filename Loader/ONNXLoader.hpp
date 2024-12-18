#include <fstream>
#include <memory>
#include <string>
#include <iostream>
#include <unordered_map>
#include <iomanip>

#include "Model/Model.hpp"
#include "Layers/Linear/Linear.hpp"
#include "Layers/ReLU/ReLU.hpp"
#include "Layers/Softmax/Softmax.hpp"
#include "Layers/Dropout/Dropout.hpp"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

class ONNXLoader {
private:
    static void loadLinearLayer(std::vector<std::unique_ptr<Layer>>& layers,
                              const onnx::TensorProto* weight_tensor,
                              const onnx::TensorProto* bias_tensor,
                              bool useGPU) {
        if (weight_tensor->dims_size() != 2) {
            throw std::runtime_error("Weight tensor must be 2-dimensional");
        }

        long output_size = weight_tensor->dims(0);
        long input_size = weight_tensor->dims(1);

        std::cout << "Linear Layer -> Input size: " << input_size;
        std::cout << ", Output size: " << output_size << "\n";

        auto layer = std::make_unique<Linear>(input_size, output_size, false); // Start on CPU

        if (weight_tensor->data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported weight data type");
        }

        const auto* weight_data = reinterpret_cast<const float*>(weight_tensor->raw_data().data());
        for (long i = 0; i < output_size; ++i) {
            for (long j = 0; j < input_size; ++j) {
                layer->params.weights[j][i] = weight_data[i * input_size + j];
            }
        }

        // Load biases
        if (bias_tensor->data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported bias data type");
        }

        const auto* bias_data = reinterpret_cast<const float*>(bias_tensor->raw_data().data());
        for (long i = 0; i < output_size; ++i) {
            layer->params.biases[0][i] = bias_data[i];
        }

        // Switch to GPU if needed
        if (useGPU) {
            layer->switchDevice(true);
        }

        layers.push_back(std::move(layer));
    }

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

        std::cout << "\n======= Loading ONNX model =======" << "\n";

        const auto& graph = model_proto.graph();
        std::vector<std::unique_ptr<Layer>> layers;

        // Store initialized tensors
        std::unordered_map<std::string, const onnx::TensorProto*> initializers;
        for (const auto& initializer : graph.initializer()) {
            initializers[initializer.name()] = &initializer;
            std::cout << "Found initializer: " << initializer.name()
                     << " with shape [" << initializer.dims(0) << ", "
                     << (initializer.dims_size() > 1 ? std::to_string(initializer.dims(1)) : "1")
                     << "]" << std::endl;
        }

        for (const auto& node : graph.node()) {
            std::cout << "Processing node: " << node.op_type() << std::endl;
            if (node.op_type() == "Gemm" || node.op_type() == "Linear") {
                if (node.input_size() < 3) {
                    throw std::runtime_error("Linear layer node missing inputs");
                }
                const auto* weight_tensor = initializers[node.input(1)];
                const auto* bias_tensor = initializers[node.input(2)];
                if (!weight_tensor || !bias_tensor) {
                    throw std::runtime_error("Could not find weights or biases for Linear layer");
                }
                loadLinearLayer(layers, weight_tensor, bias_tensor, useGPU);
            }
            else if (node.op_type() == "Relu") {
                std::cout << "Adding ReLU layer\n";
                layers.push_back(std::make_unique<ReLU>(0, 0, useGPU));
            }
            else if (node.op_type() == "Dropout") {
                float ratio = 0.2f;
                for (const auto& attr : node.attribute()) {
                    if (attr.name() == "ratio") {
                        ratio = attr.f();
                    }
                }
                std::cout << "Adding Dropout layer with ratio " << ratio << "\n";
                layers.push_back(std::make_unique<Dropout>(ratio, 0, 0, useGPU));
            }
            else if (node.op_type() == "Softmax") {
                std::cout << "Adding Softmax layer\n";
                layers.push_back(std::make_unique<Softmax>(true, 0, 0, useGPU));
            }
        }

        return Model(std::move(layers));
    }
};