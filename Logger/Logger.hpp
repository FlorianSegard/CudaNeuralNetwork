#pragma once
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
    static void debugTensor(LogLevel log_level, T& tensor) {
        if (level == log_level) {
            tensor.switchDevice(false).print();
        }
    }
};