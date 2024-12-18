#include "BinCrossEntropy.hpp"

__global__ void binaryCrossEntropyKernel(const float* predictions, const float* targets,
                                        float* gradients, float* total_loss,
                                        int width, int height,
                                        size_t predStride, size_t targetStride, size_t gradStride) {
    extern __shared__ float shared_loss[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - epsilon;
    float thread_loss = 0.0f;

    if (x < width && y < height) {
        int predIdx = x + (predStride / sizeof(float)) * y;
        int targetIdx = x + (targetStride / sizeof(float)) * y;
        int gradIdx = x + (gradStride / sizeof(float)) * y;

        // Clip predictions to prevent log(0)
        float pred = fminf(fmaxf(predictions[predIdx], epsilon), one_minus_epsilon);
        float target = targets[targetIdx];

        // Compute loss: -[t*log(p) + (1-t)*log(1-p)]
        thread_loss = -(target * logf(pred) + (1.0f - target) * logf(1.0f - pred));

        // Compute gradient: -(t/p - (1-t)/(1-p))
        gradients[gradIdx] = -(target / pred - (1.0f - target) / (1.0f - pred));
    }

    shared_loss[tid] = thread_loss;
    __syncthreads();

    // parallel reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }

    // first thread in block -> write result to global memory
    if (tid == 0) {
        atomicAdd(total_loss, shared_loss[0]);
    }
}

float computeBinaryCrossEntropyGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float* d_total_loss;
    cudaMalloc(&d_total_loss, sizeof(float));
    cudaMemset(d_total_loss, 0, sizeof(float));

    dim3 blockSize(32, 32);  // Reduced block size to allow more shared memory per block
    dim3 numBlocks((predictions.width + blockSize.x - 1) / blockSize.x,
                   (predictions.height + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(float);

    binaryCrossEntropyKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        predictions.buffer, targets.buffer, gradients.buffer, d_total_loss,
        predictions.width, predictions.height,
        predictions.stride, targets.stride, gradients.stride
    );

    float total_loss;
    cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_total_loss);

    return total_loss / static_cast<float>(predictions.width * predictions.height);
}