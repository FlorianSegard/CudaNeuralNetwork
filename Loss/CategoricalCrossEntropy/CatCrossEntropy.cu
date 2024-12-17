#include "CatCrossEntropy.hpp"

__global__ void crossEntropyLossKernel(const float* predictions, const float* targets,
                                       float* gradients, float* totalLoss,
                                       int width, int height,
                                       size_t predStride, size_t targetStride, size_t gradStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int predIdx = x + (predStride / sizeof(float)) * y;
        int targetIdx = x + (targetStride / sizeof(float)) * y;
        int gradIdx = x + (gradStride / sizeof(float)) * y;

        float pred = predictions[predIdx];
        float target = targets[targetIdx];

        // The gradient for softmax + cross-entropy is simply pred - target
        gradients[gradIdx] = pred - target;

        // Compute loss: -target * log(pred)
        // Add small epsilon to avoid log(0)
        const float epsilon = 1e-7f;
        float loss = 0.0f;
        if (target > 0) {
            loss = -target * logf(pred + epsilon);
        }

        atomicAdd(totalLoss, loss);
    }
}

float computeCatCrossEntropyLossGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    dim3 blockSize(32, 32);
    dim3 numBlocks((predictions.width + blockSize.x - 1) / blockSize.x,
                   (predictions.height + blockSize.y - 1) / blockSize.y);

    crossEntropyLossKernel<<<numBlocks, blockSize>>>(
        predictions.buffer, targets.buffer, gradients.buffer, d_loss,
        predictions.width, predictions.height,
        predictions.stride, targets.stride, gradients.stride
    );

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);
    cudaDeviceSynchronize();

    return h_loss / (float) predictions.height;
}