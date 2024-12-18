#include "CatCrossEntropy.hpp"

__global__ void crossEntropyLossKernel(const float* predictions, const float* targets,
                                      float* gradients, float* total_loss,
                                      int width, int height,
                                      size_t predStride, size_t targetStride, size_t gradStride) {
    extern __shared__ float shared_loss[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float thread_loss = 0.0f;
    const float epsilon = 1e-7f;

    if (x < width && y < height) {
        int predIdx = x + (predStride / sizeof(float)) * y;
        int targetIdx = x + (targetStride / sizeof(float)) * y;
        int gradIdx = x + (gradStride / sizeof(float)) * y;

        float pred = predictions[predIdx];
        float target = targets[targetIdx];

        // The gradient for softmax + cross-entropy
        gradients[gradIdx] = pred - target;

        // loss only for positive targets
        if (target > 0) {
            thread_loss = -target * logf(pred + epsilon);
        }
    }

    shared_loss[tid] = thread_loss;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }

    // First thread in block -> writes result to global memory
    if (tid == 0) {
        atomicAdd(total_loss, shared_loss[0]);
    }
}

float computeCatCrossEntropyLossGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float* d_total_loss;
    cudaMalloc(&d_total_loss, sizeof(float));
    cudaMemset(d_total_loss, 0, sizeof(float));

    dim3 blockSize(16, 16);  // Reduced block size to allow more shared memory per block
    dim3 numBlocks((predictions.width + blockSize.x - 1) / blockSize.x,
                   (predictions.height + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);

    crossEntropyLossKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        predictions.buffer, targets.buffer, gradients.buffer, d_total_loss,
        predictions.width, predictions.height,
        predictions.stride, targets.stride, gradients.stride
    );

    float total_loss;
    cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_total_loss);

    return total_loss / static_cast<float>(predictions.height);
}