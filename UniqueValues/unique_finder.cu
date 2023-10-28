#include "unique_finder.cuh"
#include <cuda_runtime.h>
#include <iostream>

enum { BLOCK_SIZE = 1024 };

template <typename T>
__global__ void countOccurrences(T *data, T *unique_vals, T *histogram, int n, int unique_values_) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= unique_values_) return;  // Ensure we only process for unique values

    T unique_value = unique_vals[idx];

    // Count the occurrences of unique_value in the assigned data block
    int local_count = 0;
    for (int i = blockIdx.x * blockDim.x; i < (blockIdx.x + 1) * blockDim.x && i < n; i++) {
        if (data[i] == unique_value) {
            local_count++;
        }
    }

    // Use atomic add to safely update the histogram
    atomicAdd(&histogram[unique_value], local_count);
}

template <typename T>
UniqueFinder<T>::UniqueFinder(std::vector<T>&& data, size_t nunique): unique_values_(nunique) {
    // Memory allocations and data copying
    cudaMalloc(&d_data_, data.size() * sizeof(T));
    cudaMalloc(&d_unique_values_, unique_values_ * sizeof(T));
    cudaMalloc(&d_histogram_, unique_values_ * sizeof(T));

    histogram_.resize(unique_values_, 0);
    std::vector<T> unique_values(unique_values_);
    for (int i = 0; i < unique_values_; i++) {
        unique_values[i] = static_cast<T>(i);
    }

    cudaMemcpy(d_data_, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unique_values_, unique_values.data(), unique_values_ * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram_, histogram_.data(), unique_values_ * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
UniqueFinder<T>::~UniqueFinder() {
    // Cleanup
    cudaFree(d_data_);
    cudaFree(d_unique_values_);
    cudaFree(d_histogram_);
}

template <typename T>
std::vector<T> UniqueFinder<T>::findUnique() {
    int blocksPerGrid = (unique_values_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    countOccurrences<<<blocksPerGrid, BLOCK_SIZE>>>(d_data_, d_unique_values_, d_histogram_, unique_values_, unique_values_);

    cudaMemcpy(histogram_.data(), d_histogram_, unique_values_ * sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<T> unique_elements;
    for (int i = 0; i < unique_values_; i++) {
        if (histogram_[i] == 1) {
            unique_elements.push_back(static_cast<T>(i));
        }
    }

    return unique_elements;
}
