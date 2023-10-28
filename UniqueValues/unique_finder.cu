#include "unique_finder.cuh"
#include <cuda_runtime.h>

// Define block size
enum { BLOCK_SIZE = 256 };

// Kernel to compute the histogram
template <typename T>
__global__ void histogramKernel(const T* data, T* histogram, size_t dataSize, size_t nunique) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the index is within the bounds
    if (idx < dataSize) {
        T value = data[idx];
        for (int i = 0; i < nunique; ++i) {
            if (value == i) {
                atomicAdd(&histogram[i], 1);
                break;
            }
        }
    }
}

// Kernel to find unique values
template <typename T>
__global__ void findUniqueKernel(const T* histogram, T* output, size_t nunique) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the index is within the bounds
    if (idx < nunique) {
        if (histogram[idx] == 1) {
            output[idx] = idx;
        } else {
            output[idx] = -1;  // A flag to indicate non-unique value
        }
    }
}

template <typename T>
UniqueFinder<T>::UniqueFinder(const std::vector<T>& data, size_t nunique) : nunique_(nunique), data_size_(data.size()) {
    // Allocate device memory
    cudaMalloc((void**)&d_data, data_size_ * sizeof(T));
    cudaMalloc((void**)&d_histogram, nunique_ * sizeof(T));

    // Copy data to device
    cudaMemcpy(d_data, data.data(), data_size_ * sizeof(T), cudaMemcpyHostToDevice);

    // Initialize histogram to zeros
    cudaMemset(d_histogram, 0, nunique_ * sizeof(T));
}

template <typename T>
UniqueFinder<T>::~UniqueFinder() {
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);
}

template <typename T>
std::vector<T> UniqueFinder<T>::find_unique() {
    // Compute the histogram
    int gridSize = (data_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramKernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_histogram, data_size_, nunique_);

    // Find unique values
    T* d_output;
    cudaMalloc((void**)&d_output, nunique_ * sizeof(T));
    findUniqueKernel<<<gridSize, BLOCK_SIZE>>>(d_histogram, d_output, nunique_);

    // Copy the result back to host
    std::vector<T> unique_values(nunique_);
    cudaMemcpy(unique_values.data(), d_output, nunique_ * sizeof(T), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_output);

    // Filter out non-unique values
    unique_values.erase(std::remove(unique_values.begin(), unique_values.end(), -1), unique_values.end());

    return unique_values;
}
