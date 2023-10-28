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
__global__ void filterUniqueValues(T *histogram, T *unique_values, T *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && histogram[idx] == 1) {
        // Use atomicAdd to get a unique index in the output array
        int pos = atomicAdd(&output[0], 1);
        output[pos + 1] = idx;  // +1 because output[0] is used for count
    }
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

    T *d_unique_elements;
    int max_unique = unique_values_;  // In the worst case, all values are unique
    cudaMalloc(&d_unique_elements, (max_unique + 1) * sizeof(T));  // +1 to store the count at the beginning
    cudaMemset(d_unique_elements, 0, sizeof(T));  // Set count to 0

    filterUniqueValues<<<blocksPerGrid, BLOCK_SIZE>>>(d_histogram_, d_unique_values_, d_unique_elements, unique_values_);

    int unique_count;
    cudaMemcpy(&unique_count, d_unique_elements, sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<T> unique_elements(unique_count);
    cudaMemcpy(unique_elements.data(), d_unique_elements + 1, unique_count * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_unique_elements);

    return unique_elements;
}
