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

__global__ void prefix_sum_first_pass(int* input, int* output, int* blockSums, int n) {
    extern __shared__ int temp[];

    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int offset = 1;

    int idx = 2 * threadId + blockId * 2 * blockDim.x;
    int idxNext = idx + 1;

    // Load input into shared memory
    temp[2 * threadId] = (idx < n) ? input[idx] : 0;
    temp[2 * threadId + 1] = (idxNext < n) ? input[idxNext] : 0;
    __syncthreads();

    // Up-sweep phase (reduce)
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (threadId == 0) {
        temp[2 * blockDim.x - 1] = 0;
    }

    // Down-sweep phase (post-reduce)
    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results to output
    if (idx < n) {
        output[idx] = temp[2 * threadId];
    }
    if (idxNext < n) {
        output[idxNext] = temp[2 * threadId + 1];
    }

    // Write the block's sum to blockSums
    if (threadId == 0) {
        blockSums[blockId] = temp[2 * blockDim.x - 2] + ((blockId * 2 * blockDim.x + 2 * blockDim.x - 1) < n ? input[blockId * 2 * blockDim.x + 2 * blockDim.x - 1] : 0);
    }
}

__global__ void add_block_sums(int* input, int* blockSums, int n) {
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    if (globalId < n) {
        input[globalId] += blockSums[blockIdx.x];
    }
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

void recursive_prefix_sum(int* d_input, int* d_output, int* d_blockSums, int length) {
    int numBlocks = (length + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    if (numBlocks > 1) {
        // If numBlocks is greater than BLOCK_SIZE, then we need another layer of block sums
        int* d_nextBlockSums;
        cudaMalloc(&d_nextBlockSums, numBlocks * sizeof(int));
        recursive_prefix_sum(d_blockSums, d_blockSums, d_nextBlockSums, numBlocks);
        cudaFree(d_nextBlockSums);
    }
    prefix_sum_first_pass<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * 2 * sizeof(int)>>>(d_input, d_output, d_blockSums, length);
    cudaDeviceSynchronize();
    add_block_sums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, length);
    cudaDeviceSynchronize();
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
    int numBlocks = (nunique + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    int* d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    recursive_prefix_sum(d_binary, d_prefix_sum, d_blockSums, nunique);
    cudaFree(d_blockSums);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after launching [add_block_sums]: %s\n", cudaGetErrorString(err));
    }

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
