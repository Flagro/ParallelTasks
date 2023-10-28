#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

enum { BLOCK_SIZE = 256 };

__global__ void count_occurrences_kernel(int* data, int* global_histogram, int n, int nunique, int chunk_size) {
    extern __shared__ int local_histogram[];

    int threadId = threadIdx.x;
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize local histogram in shared memory
    for (int i = threadId; i < nunique; i += blockDim.x) {
        local_histogram[i] = 0;
    }
    __syncthreads();

    // Each thread processes a chunk of data and updates the local histogram
    for (int i = 0; i < chunk_size; i++) {
        int dataIdx = globalId * chunk_size + i;
        if (dataIdx < n) {
            atomicAdd(&local_histogram[data[dataIdx]], 1);
        }
    }
    __syncthreads();

    // Update global histogram from local histograms
    for (int i = threadId; i < nunique; i += blockDim.x) {
        atomicAdd(&global_histogram[i], local_histogram[i]);
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
    int chunk_size = (n + blocks_count * BLOCK_SIZE - 1) / (blocks_count * BLOCK_SIZE);
    count_occurrences_kernel<<<blocks_count, BLOCK_SIZE, nunique * sizeof(int)>>>(d_data, d_histogram, n, nunique, chunk_size);

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
