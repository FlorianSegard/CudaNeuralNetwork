#pragma once
#include "../Layers/Layers.hpp"
#include "ModelLoader.hpp"
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <onnx/onnx-ml.pb.h>

class ModelLoader {
public:
    static void loadONNX(const std::string& model_path, bool useGPU = false) {
        onnx::ModelProto model;
        std::ifstream in(model_path, std::ios_base::binary);
        std::cout << "parsing model" << "\n";
        model.ParseFromIstream(&in);
        in.close();
        std::cout << "model size:";
        std::cout << model.graph().input().size() << "\n";
    }
};