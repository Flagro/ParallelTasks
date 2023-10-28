#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

enum { BLOCK_SIZE = 256, CHUNK_SIZE = 512 };

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

__global__ void histogram_to_binary(int* histogram, int* binary, int nunique) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nunique) {
        binary[index] = (histogram[index] == 1) ? 1 : 0;
    }
}

__global__ void prefix_sum_kernel(int* input, int* output, int n) {
    extern __shared__ int temp[];  // allocated on invocation

    int threadId = threadIdx.x;
    int pout = 0, pin = 1;

    // Load input into shared memory.
    // This is exclusive scan, so shift right by one and set first element to 0
    temp[pout*n + threadId] = (threadId > 0) ? input[threadId-1] : 0;
    __syncthreads();

    for(int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;  // double buffer
        pin = 1 - pout;
        if (threadId >= offset) {
            temp[pout*n+threadId] += temp[pin*n+threadId - offset];
        } else {
            temp[pout*n+threadId] = temp[pin*n+threadId];
        }
        __syncthreads();
    }
    output[threadId] = temp[pout*n+threadId];  // write output
}

__global__ void extract_unique_values(int* histogram, int* prefixSum, int* data, int* unique_values, int nunique) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nunique) {
        if (histogram[index] == 1) {
            unique_values[prefixSum[index] - 1] = data[index];
        }
    }
}

int nextPowerOf2(int n) {
    int count = 0;
    if (n && !(n & (n - 1))) {
        return n;
    }
    while(n != 0) {
        n >>= 1;
        count += 1;
    }
    return 1 << count;
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

    // Obtain the histogram of the data
    int blocks_count = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    count_occurrences_kernel<<<blocks_count, BLOCK_SIZE, nunique * sizeof(int)>>>(d_data, d_histogram, n, nunique, CHUNK_SIZE);

    // Convert histogram to binary format
    int* d_binary;
    cudaMalloc(&d_binary, nunique * sizeof(int));
    histogram_to_binary<<<(nunique + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_histogram, d_binary, nunique);

    // Allocate memory for prefix_sum and unique_values on the device
    int* d_prefix_sum, *d_unique_values;
    cudaMalloc(&d_prefix_sum, nunique * sizeof(int));
    cudaMalloc(&d_unique_values, nunique * sizeof(int));

    // Compute prefix sum
    int padded_size = nextPowerOf2(nunique);
    int shared_mem_size = 2 * padded_size * sizeof(int);
    prefix_sum_kernel<<<1, padded_size, shared_mem_size>>>(d_histogram, d_prefix_sum, padded_size);
    // After computing prefix sum
    int* h_prefix_sum_debug = new int[nunique];
    cudaMemcpy(h_prefix_sum_debug, d_prefix_sum, nunique * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nunique; i++) {
        std::cout << "PrefixSum[" << i << "]: " << h_prefix_sum_debug[i] << std::endl;
    }
    delete[] h_prefix_sum_debug;




    // Extract unique values based on the prefix sum
    extract_unique_values<<<(nunique + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_histogram, d_binary, d_data, d_unique_values, nunique);

    // 1. Get the number of unique values
    int num_unique;
    cudaMemcpy(&num_unique, &d_binary[nunique - 1], sizeof(int), cudaMemcpyDeviceToHost);

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
    // delete[] h_histogram;
    // delete[] h_unique_values;

    return unique_elements;
}
