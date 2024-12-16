#pragma once
#include <cfloat>
#include <iostream>

enum class LogLevel {
    NONE = 0,
    INFER = 1,
    BACKPROP = 2,
    LOSS = 3,
    DEBUG = 4,
    ALL = 5
};

class Logger {
private:
    static LogLevel level;

public:
    static void setLevel(LogLevel l) { level = l; }
    static LogLevel getLevel() { return level; }

    template<typename T>
    static void infer(const T& message) {
        if (level == LogLevel::INFER || level == LogLevel::ALL) {
            std::cout << message << std::endl;
        }
    }

    template<typename T>
    static void backprop(const T& message) {
        if (level == LogLevel::BACKPROP || level == LogLevel::ALL) {
            std::cout << message << std::endl;
        }
    }

    template<typename T>
    static void loss(const T& message) {
        if (level == LogLevel::LOSS || level == LogLevel::ALL) {
            std::cout << message << std::endl;
        }
    }

    template<typename T>
    static void debug(const T& message) {
        if (level >= LogLevel::DEBUG) {
            std::cout << message << std::endl;
        }
    }

    template<typename T>
    static void debugTensor(LogLevel log_level, T& tensor, bool print_value = false) {
        if (level == log_level) {
            auto cpu = tensor.switchDevice(false);
            float maxVal = -FLT_MAX;
            float minVal = FLT_MAX;
            float sumAbs = 0.0f;

            for (int i = 0; i < cpu.height; i++) {
                for (int j = 0; j < cpu.width; j++) {
                    float val = cpu[i][j];
                    maxVal = std::max(maxVal, val);
                    minVal = std::min(minVal, val);
                    sumAbs += std::abs(val);
                }
            }
            if (print_value)
                cpu.print();

            std::cout << "| Height=" << cpu.height << " && Width=" << cpu.width << std::endl;
            std::cout  << "| -> stats - min: " << minVal
                      << " max: " << maxVal
                      << " mean abs: " << sumAbs/(cpu.height*cpu.width) << std::endl;
        }
    }
};