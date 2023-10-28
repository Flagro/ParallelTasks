#include "unique_finder.cuh"
#include <cuda_runtime.h>
#include <iostream>

enum { BLOCK_SIZE = 32 };

template <typename T>
__global__ void countOccurrences(T *data, T *unique_vals, T *histogram, int n, int unique_values_) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;  // Ensure we only process valid data indices

    T value = data[idx];
    for (int u = 0; u < unique_values_; u++) {
        if (value == unique_vals[u]) {
            atomicAdd(&histogram[u], 1);
            break; // Break once we've found a match
        }
    }
}


template <typename T>
__global__ void extractUniqueValues(T *unique_vals, T *histogram, T *output, int *output_count, int unique_values_) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= unique_values_) return;

    if (histogram[idx] == 1) {
        int pos = atomicAdd(output_count, 1);
        output[pos] = unique_vals[idx];
    }
}

template <typename T>
UniqueFinder<T>::UniqueFinder(const std::vector<T>& data, size_t nunique): unique_values_(nunique) {
    // Memory allocations and data copying
    cudaMalloc(&d_data_, data.size() * sizeof(T));
    cudaMalloc(&d_unique_values_, unique_values_ * sizeof(T));
    cudaMalloc(&d_histogram_, unique_values_ * sizeof(T));
    cudaMalloc(&d_output_count_, sizeof(int));

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
    cudaFree(d_output_count_);
}

template <typename T>
std::vector<T> UniqueFinder<T>::findUnique() {
    int blocksPerGrid = (data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    countOccurrences<<<blocksPerGrid, BLOCK_SIZE>>>(d_data_, d_unique_values_, d_histogram_, data.size(), unique_values_);

    // Debug: Check histogram
    std::vector<T> debug_histogram(unique_values_);
    cudaMemcpy(debug_histogram.data(), d_histogram_, unique_values_ * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "Histogram: ";
    for (int i = 0; i < unique_values_; i++) {
        std::cout << debug_histogram[i] << " ";
    }
    std::cout << std::endl;

    int h_output_count = 0;
    cudaMemcpy(d_output_count_, &h_output_count, sizeof(int), cudaMemcpyHostToDevice);
    
    extractUniqueValues<<<blocksPerGrid, BLOCK_SIZE>>>(d_unique_values_, d_histogram_, d_data_, d_output_count_, unique_values_);

    // Debug: Check unique values and count
    cudaMemcpy(&h_output_count, d_output_count_, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<T> debug_output(h_output_count);
    cudaMemcpy(debug_output.data(), d_data_, h_output_count * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "Unique Values (" << h_output_count << "): ";
    for (int i = 0; i < h_output_count; i++) {
        std::cout << debug_output[i] << " ";
    }
    std::cout << std::endl;

    return debug_output;
}

