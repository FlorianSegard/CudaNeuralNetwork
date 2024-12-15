#include "Loss.hpp"

__global__ void sumOfSquaresKernel(const float* input, int width, int height, size_t stride, float* partialSums) {
    int pitch = stride / sizeof(float);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    // Local sum for this thread
    float localSum = (float)0;

    // Stride through the entire tensor, covering different elements
    // Each thread handles multiple elements.
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int y = i / width;
        int x = i % width;

        int input_index = y * pitch + x;
        float val = input[input_index];
        localSum += val * val;
    }

    // Use shared memory for intra-block reduction
    extern __shared__ float shared[];
    int tx = threadIdx.x;
    shared[tx] = localSum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            shared[tx] += shared[tx + s];
        }
        __syncthreads();
    }

    // Add the block's result to the global partial sum
    if (tx == 0) {
        atomicAdd(partialSums, shared[0]);
    }
}

// Templated host function to invoke the kernel and compute sum of squares on the GPU
float sumOfSquaresGPU(const Tensor<float>& input) {
    // Allocate device memory for the partial sum result
    float* d_partialSum;
    cudaMalloc(&d_partialSum, sizeof(float));
    cudaMemset(d_partialSum, 0, sizeof(float));

    // Launch configuration
    int blockSize = 256;
    int gridSize = (input.width * input.height + blockSize - 1) / blockSize;

    // Dynamic shared memory: we pass blockSize * sizeof(T)
    sumOfSquaresKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(input.buffer, input.width, input.height, input.stride, d_partialSum);
    cudaDeviceSynchronize();

    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_partialSum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partialSum);

    return result;
}



__global__ void crossEntropyLossKernel(const float* predictions, const float* targets,
                                      float* gradients, float* losses,
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
        float loss = -target * logf(pred + epsilon);
        losses[y * width + x] = loss;
    }
}

float computeCrossEntropyLossGPU(const Tensor<float>& predictions, const Tensor<float>& targets, Tensor<float>& gradients) {
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    dim3 blockSize(16, 16);
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

    return h_loss / (float) predictions.height;
}