#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

enum { BLOCK_SIZE = 1024, CHUNK_SIZE = 128000 };

/*
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
*/

__global__ void count_occurrences_kernel(int* data, int* histogram, int n, int nunique, int chunk_size) {
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread processes a chunk of data and updates the histogram directly
    for (int i = 0; i < chunk_size; i++) {
        int dataIdx = globalId * chunk_size + i;
        if (dataIdx < n) {
            atomicAdd(&histogram[data[dataIdx]], 1);
        }
    }
}

__global__ void histogram_to_binary(int* histogram, int* binary, int nunique) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nunique) {
        binary[index] = (histogram[index] == 1) ? 1 : 0;
    }
}

__global__ void simple_prefix_sum(int* input, int* output, int n) {
    extern __shared__ int temp[];

    int threadId = threadIdx.x;

    // Load input into shared memory.
    temp[threadId] = (threadId < n) ? input[threadId] : 0;
    __syncthreads();

    for (int stride = 1; stride < n; stride *= 2) {
        int value = 0;
        if (threadId >= stride) {
            value = temp[threadId - stride];
        }
        __syncthreads();
        temp[threadId] += value;
        __syncthreads();
    }

    if (threadId < n) {
        output[threadId] = temp[threadId];
    }
}

__global__ void extract_unique_values(int* histogram, int* prefixSum, int* unique_values, int nunique) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nunique) {
        if (histogram[index] == 1) {
            unique_values[prefixSum[index] - 1] = index;
        }
    }
}

UniqueFinder::UniqueFinder(const std::vector<int>& data, int nunique) : data(data), unique_values(nunique) {}

std::vector<int> UniqueFinder::find_unique() {
    int n = this->data.size();
    int nunique = this->unique_values;

    int* d_data;
    int* d_histogram;
    cudaError_t err;

    err = cudaMalloc(&d_data, n * sizeof(int));
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMalloc(&d_histogram, nunique * sizeof(int));
    cudaDeviceSynchronize();
    err = cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy to a device memory: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemset(d_histogram, 0, nunique * sizeof(int));
    cudaDeviceSynchronize();

    // Obtain the histogram of the data
    int blocks_count = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    count_occurrences_kernel<<<blocks_count, BLOCK_SIZE, nunique * sizeof(int)>>>(d_data, d_histogram, n, nunique, CHUNK_SIZE);
    cudaDeviceSynchronize();
    // After generating the histogram
    int* h_histogram_debug = new int[nunique];
    cudaMemcpy(h_histogram_debug, d_histogram, nunique * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nunique; i++) {
        std::cout << "Hist[" << i << "]: " << h_histogram_debug[i] << std::endl;
    }
    delete[] h_histogram_debug;

    // Convert histogram to binary format
    int* d_binary;
    cudaMalloc(&d_binary, nunique * sizeof(int));
    histogram_to_binary<<<(nunique + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_histogram, d_binary, nunique);

    // After converting to binary
    int* h_binary_debug = new int[nunique];
    cudaMemcpy(h_binary_debug, d_binary, nunique * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nunique; i++) {
        std::cout << "Binary[" << i << "]: " << h_binary_debug[i] << std::endl;
    }
    delete[] h_binary_debug;

    // Allocate memory for prefix_sum and unique_values on the device
    int* d_prefix_sum, *d_unique_values;
    cudaMalloc(&d_prefix_sum, nunique * sizeof(int));
    cudaMalloc(&d_unique_values, nunique * sizeof(int));

    // Compute prefix sum
    int blockSize = min(nunique, BLOCK_SIZE);
    simple_prefix_sum<<<1, blockSize, blockSize * sizeof(int)>>>(d_binary, d_prefix_sum, nunique);

    // After computing prefix sum
    int* h_prefix_sum_debug = new int[nunique];
    cudaMemcpy(h_prefix_sum_debug, d_prefix_sum, nunique * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nunique; i++) {
        std::cout << "PrefixSum[" << i << "]: " << h_prefix_sum_debug[i] << std::endl;
    }
    delete[] h_prefix_sum_debug;

    // Extract unique values based on the prefix sum
    extract_unique_values<<<(nunique + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_histogram, d_prefix_sum, d_unique_values, nunique);

    // 1. Get the number of unique values
    int num_unique;
    cudaMemcpy(&num_unique, &d_prefix_sum[nunique - 1], sizeof(int), cudaMemcpyDeviceToHost);

    // 2. Allocate space for these unique values on the host
    std::vector<int> unique_elements(num_unique);

    // 3. Copy the unique values from the device to the host memory
    cudaMemcpy(unique_elements.data(), d_unique_values, num_unique * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_prefix_sum);
    cudaFree(d_binary);
    cudaFree(d_unique_values);

    return unique_elements;
}
