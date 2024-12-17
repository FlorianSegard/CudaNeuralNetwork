#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "../Tensor/Tensor.hpp"

class TabularLoader {
private:
    static std::vector<std::string> splitLine(const std::string& line, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

public:
    static std::pair<std::vector<Tensor<float>>, std::vector<float>> loadCSV(
            const std::string& filepath,
            char delimiter = ',',
            bool normalize = false,
            int max_rows = -1,
            bool skip_header = true) {

        std::vector<Tensor<float>> features;
        std::vector<float> labels;
        std::vector<float> min_vals;
        std::vector<float> max_vals;
        int num_features = 0;

        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::string line;
        bool first_data_row = true;

        // Skip header if requested
        if (skip_header && std::getline(file, line)) {
            num_features = splitLine(line, delimiter).size() - 1;  // -1 for label column
        }

        // First pass: collect statistics for normalization
        if (normalize) {
            min_vals.resize(num_features, std::numeric_limits<float>::max());
            max_vals.resize(num_features, std::numeric_limits<float>::lowest());

            while (std::getline(file, line)) {
                auto tokens = splitLine(line, delimiter);
                for (int i = 0; i < num_features; ++i) {
                    float val = std::stof(tokens[i]);
                    min_vals[i] = std::min(min_vals[i], val);
                    max_vals[i] = std::max(max_vals[i], val);
                }
            }
            file.clear();
            file.seekg(0);
            if (skip_header) std::getline(file, line);  // Skip header again
        }

        // Second pass: read data
        int row_count = 0;
        while (std::getline(file, line) && (max_rows == -1 || row_count < max_rows)) {
            auto tokens = splitLine(line, delimiter);

            if (first_data_row) {
                num_features = tokens.size() - 1;  // -1 for label column
                first_data_row = false;
            }

            // Create tensor for features
            Tensor<float> feature_tensor(num_features, 1, false);
            for (int i = 0; i < num_features; ++i) {
                float val = std::stof(tokens[i]);
                if (normalize) {
                    val = (val - min_vals[i]) / (max_vals[i] - min_vals[i]);
                }
                feature_tensor[0][i] = val;
            }
            features.push_back(std::move(feature_tensor));

            // Store label
            labels.push_back(std::stof(tokens.back()));
            row_count++;
        }

        return {std::move(features), std::move(labels)};
    }
};