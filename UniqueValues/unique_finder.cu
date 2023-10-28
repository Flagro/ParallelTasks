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
__global__ void filterUnique(T *histogram, T *output, int n, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;

    if (histogram[idx] == 1) {
        int pos = atomicAdd(count, 1);  // Safely increment the counter and get the current position
        output[pos] = idx;  // Store the unique value in the output array
    }
}

template <typename T>
UniqueFinder<T>::UniqueFinder(std::vector<T>&& data, size_t nunique): unique_values_(nunique) {
    // Memory allocations and data copying
    cudaMalloc(&d_data_, data.size() * sizeof(T));
    cudaMalloc(&d_unique_values_, unique_values_ * sizeof(T));
    cudaMalloc(&d_unique_elements_, unique_values_ * sizeof(T));
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
    cudaFree(d_unique_elements_);
    cudaFree(d_histogram_);
}

template <typename T>
std::vector<T> UniqueFinder<T>::findUnique() {
    int blocksPerGrid = (unique_values_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    countOccurrences<<<blocksPerGrid, BLOCK_SIZE>>>(d_data_, d_unique_values_, d_histogram_, unique_values_, unique_values_);

    cudaMemcpy(histogram_.data(), d_histogram_, unique_values_ * sizeof(T), cudaMemcpyDeviceToHost);

    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));  // Initialize the count to 0

    filterUnique<<<blocksPerGrid, BLOCK_SIZE>>>(d_histogram_, d_unique_elements_, unique_values_, d_count);

    cudaMemcpy(&num_unique_elements_, d_count, sizeof(int), cudaMemcpyDeviceToHost);  // Get the number of unique elements found

    std::vector<T> unique_elements(num_unique_elements_);
    cudaMemcpy(unique_elements.data(), d_unique_elements_, num_unique_elements_ * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_count);

    return unique_elements;
}
