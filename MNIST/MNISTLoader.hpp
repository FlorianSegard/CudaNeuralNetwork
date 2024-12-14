#pragma once
#include <fstream>
#include <vector>
#include <string>
#include "../Tensor/Tensor.hpp"

class MNISTLoader {
private:
    static int reverseInt(int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

public:
    static std::pair<std::vector<Tensor<float>>, std::vector<int>> loadMNIST(
            const std::string& image_path,
            const std::string& label_path,
            bool normalize = true,
            int max_images = -1) {

        std::vector<Tensor<float>> images;
        std::vector<int> labels;

        // Read images
        std::ifstream image_file(image_path, std::ios::binary);
        if (!image_file.is_open()) {
            throw std::runtime_error("Cannot open image file: " + image_path);
        }

        int magic_number = 0;
        int number_of_images = 0;
        int rows = 0;
        int cols = 0;

        image_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        image_file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        image_file.read((char*)&rows, sizeof(rows));
        rows = reverseInt(rows);

        image_file.read((char*)&cols, sizeof(cols));
        cols = reverseInt(cols);

        if (max_images > 0 && max_images < number_of_images) {
            number_of_images = max_images;
        }

        // Reserve space to avoid reallocations
        images.reserve(number_of_images);
        labels.reserve(number_of_images);

        // Read labels
        std::ifstream label_file(label_path, std::ios::binary);
        if (!label_file.is_open()) {
            throw std::runtime_error("Cannot open label file: " + label_path);
        }

        int label_magic_number = 0;
        int number_of_labels = 0;

        label_file.read((char*)&label_magic_number, sizeof(label_magic_number));
        label_magic_number = reverseInt(label_magic_number);

        label_file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        // Read image data
        std::vector<unsigned char> pixels(rows * cols);
        for (int i = 0; i < number_of_images; i++) {
            // Read all pixels for current image
            image_file.read(reinterpret_cast<char*>(pixels.data()), rows * cols);

            // Create new tensor for the image and fill it
            images.emplace_back(784, 1, false);  // Construct tensor in-place
            for (int j = 0; j < rows * cols; j++) {
                images.back()[0][j] = normalize ? (float)pixels[j] / 255.0f : (float)pixels[j];
            }

            // Read corresponding label
            unsigned char label = 0;
            label_file.read((char*)&label, sizeof(label));
            labels.push_back((int)label);
        }

        return {std::move(images), std::move(labels)};  // Use move semantics
    }

    // Utility function to convert label to one-hot encoded tensor
    static Tensor<float> labelToOneHot(int label, int num_classes = 10) {
        Tensor<float> one_hot(num_classes, 1, false);
        one_hot.fillZero();
        one_hot[0][label] = 1.0f;
        return one_hot;
    }
};