#include "BinCrossEntropy.hpp"

__global__ void binaryCrossEntropyKernel(const float* predictions, const float* targets,
                                        float* gradients, float* losses,
                                        int width, int height,
                                        size_t predStride, size_t targetStride, size_t gradStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int predIdx = x + (predStride / sizeof(float)) * y;
        int targetIdx = x + (targetStride / sizeof(float)) * y;
        int gradIdx = x + (gradStride / sizeof(float)) * y;

        const float epsilon = 1e-7f;
        const float one_minus_epsilon = 1.0f - epsilon;

        // Clip predictions to prevent log(0)
        float pred = fminf(fmaxf(predictions[predIdx], epsilon), one_minus_epsilon);
        float target = targets[targetIdx];

        // Compute loss: -[t*log(p) + (1-t)*log(1-p)]
        float loss = -(target * logf(pred) + (1.0f - target) * logf(1.0f - pred));

        // Compute gradient: -(t/p - (1-t)/(1-p))
        gradients[gradIdx] = -(target / pred - (1.0f - target) / (1.0f - pred));

        losses[y * width + x] = loss;
    }
}

float computeBinaryCrossEntropyGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float* d_losses;
    cudaMalloc(&d_losses, predictions.width * predictions.height * sizeof(float));

    dim3 blockSize(32, 32);
    dim3 numBlocks((predictions.width + blockSize.x - 1) / blockSize.x,
                   (predictions.height + blockSize.y - 1) / blockSize.y);

    binaryCrossEntropyKernel<<<numBlocks, blockSize>>>(
        predictions.buffer, targets.buffer, gradients.buffer, d_losses,
        predictions.width, predictions.height,
        predictions.stride, targets.stride, gradients.stride
    );

    // Compute average loss
    float total_loss = 0.0f;
    int totalElements = predictions.width * predictions.height;
    float* h_losses = new float[totalElements];

    cudaMemcpy(h_losses, d_losses, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < totalElements; i++) {
        total_loss += h_losses[i];
    }

    delete[] h_losses;
    cudaFree(d_losses);

    return total_loss / static_cast<float>(totalElements);
}