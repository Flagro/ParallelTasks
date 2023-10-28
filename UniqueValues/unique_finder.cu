#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

enum { BLOCK_SIZE = 256 };

__global__ void count_occurrences_kernel(int* data, int* histogram, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&histogram[data[index]], 1);
    }
}

UniqueFinder::UniqueFinder(const std::vector<int>& data, int nunique) {
    this->data = data;
    this->unique_values = nunique;
}

UniqueFinder::~UniqueFinder() {
}

std::vector<int> UniqueFinder::find_unique() {
    int n = this->data.size();
    int nunique = this->unique_values;

    int* d_data;
    int* d_histogram;
    
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_histogram, nunique * sizeof(int));
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, nunique * sizeof(int));

    int blocks_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_occurrences_kernel<<<blocks, BLOCK_SIZE>>>(d_data, d_histogram, n);

    int* h_histogram = new int[nunique];
    cudaMemcpy(h_histogram, d_histogram, nunique * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> unique_elements;
    for (int i = 0; i < nunique; i++) {
        if (h_histogram[i] == 1) {
            unique_elements.push_back(i);
        }
    }

    cudaFree(d_data);
    cudaFree(d_histogram);
    delete[] h_histogram;

    return unique_elements;
}
