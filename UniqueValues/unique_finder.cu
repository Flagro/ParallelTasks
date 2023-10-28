#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

template <typename T>
__global__ void count_occurrences_kernel(const T* data, size_t data_size, int* histogram, size_t nunique, T* unique_values) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < data_size; i += stride) {
        T value = data[i];
        if (value >= 0 && value < nunique) {
            atomicAdd(&histogram[value], 1);
        }
    }
}

template <typename T>
__global__ void find_unique_kernel(int* histogram, size_t nunique, T* unique_values, int* unique_counter) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < nunique && histogram[index] == 1) {
        int pos = atomicAdd(unique_counter, 1);
        unique_values[pos] = index;
    }
}

template <typename T>
UniqueFinder<T>::UniqueFinder(const std::vector<T>& data, size_t nunique)
    : data_size(data.size()), nunique(nunique) {

    // Allocate device memory for data, histogram, and unique values
    cudaMalloc(&d_data, data_size * sizeof(T));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_histogram, nunique * sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_unique_values, nunique * sizeof(T));
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMalloc(&d_unique_counter, sizeof(int));
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copy data to device
    cudaMemcpy(d_data, data.data(), data_size * sizeof(T), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_data, data.data(), data_size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to device: " << cudaGetErrorString(err) << std::endl;
        // Handle the error, e.g., by exiting or throwing an exception
        exit(1);
    }

    // Initialize histogram and unique counter to zero
    cudaMemset(d_histogram, 0, nunique * sizeof(int));
    cudaMemset(d_unique_counter, 0, sizeof(int));
}

template <typename T>
UniqueFinder<T>::~UniqueFinder() {
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_unique_values);
    cudaFree(d_unique_counter);
}

template <typename T>
std::vector<T> UniqueFinder<T>::find_unique() {
    int num_blocks = (data_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel to count occurrences
    count_occurrences_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, data_size, d_histogram, nunique, d_unique_values);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    // Launch kernel to find unique numbers
    find_unique_kernel<<<num_blocks, BLOCK_SIZE>>>(d_histogram, nunique, d_unique_values, d_unique_counter);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copy unique values to host
    int unique_count;
    cudaMemcpy(&unique_count, d_unique_counter, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<T> unique_values(unique_count);
    cudaMemcpy(unique_values.data(), d_unique_values, unique_count * sizeof(T), cudaMemcpyDeviceToHost);

    return unique_values;
}
