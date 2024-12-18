# CUDA Neural Network framework

A C++ neural network implementation supporting both CPU and GPU execution through CUDA. The framework provides a flexible architecture for deep learning with automatic differentiation and CUDA-accelerated tensor operations.

## Core architecture

### Tensor operations
The framework is built around an efficient tensor implementation supporting:
- CPU/GPU memory management with automatic device switching
- Basic operations: transpose, dot product, element-wise multiplication
- CUDA-optimized kernels with shared memory utilization
- Memory-aligned storage with proper striding
- Template support for different numeric types (float, double, int)

### Performance features

| Operation Type | Implementation Details |
|---------------|------------------------|
| Matrix Multiplication | Tiled algorithm with shared memory (TILE_SIZE: 32) |
| Memory Management | Pitched allocation for optimal memory access |
| Device Handling | Automatic CPU/GPU data transfer |
| Batch Processing | Vectorized operations for training efficiency |

## Components

### Neural layers
| Layer | Features                                                                                                    |
|-------|-------------------------------------------------------------------------------------------------------------|
| Linear | • Xavier initialization<br>• Configurable input/output dimensions<br>• Forward / backward pass support<br>• Batch size handling |
| ReLU | • Zero-memory activation<br>• Optimized backward pass                                                       |
| Sigmoid | • Numerically stable implementation<br>• Binary cross-entropy integration                                   |
| Softmax | • Stable computation with max subtraction<br>• Cross-entropy integration                                    |
| Dropout | • Training/eval mode switching<br>• Configurable drop rate                                                  |

### Training components
| Component | Implementation |
|-----------|----------------|
| **Optimizers** | • SGD with momentum<br>• Configurable weight decay<br>• Gradient clipping |
| **Scheduler** | • ReduceLROnPlateau<br>• Configurable patience & factor |
| **Loss Functions** | • MSE<br>• Binary Cross-Entropy<br>• Categorical Cross-Entropy |

### Data handling (in ./Examples)
- MNIST dataset loader with normalization options
- Tabular data loader for CSV files
- ONNX model import functionality (weights / biases / activation functions)

## Example usage

```cpp
// Initialize model
Model model;
model.setOptimizer(SGD(0.01f, 0.9f, 0.0001f));  // lr, momentum, weight_decay

// Add layers
model.addLayer(std::make_unique<Linear>(784, 128, true));  // GPU enabled
model.addLayer(std::make_unique<ReLU>());
model.addLayer(std::make_unique<Dropout>(0.2f));
model.addLayer(std::make_unique<Linear>(128, 10, true));
model.addLayer(std::make_unique<Softmax>(true));

// Training loop
Tensor<float> predictions = model.forward(input);
auto [loss, gradients] = CategoricalCrossEntropyLoss(predictions, targets);
model.backward(gradients);
model.step();
```

## Implemented applications

| Application | Description                                                     |
|-------------|-----------------------------------------------------------------|
| MNIST Classification | Digit recognition with dropout and learning rate scheduling     |
| Iris Classification | Multi-class flower classification                               |
| Breast Cancer Classification | Binary classification with regularization (logistic regression) |
| California Housing | Regression with multi-layer architecture                        |

## Requirements
- CUDA Toolkit
- C++17 compatible compiler
- ONNX runtime libraries

## Build and run
```bash
# Compile with GPU support
./script.sh

# Run with GPU
./output --gpu

# Available logging levels
./output --infer    # Inference logging
./output --back     # Backprop logging
./output --loss     # Loss computation logging
./output --debug    # Detailed debug information
./output --all      # All logging enabled
```