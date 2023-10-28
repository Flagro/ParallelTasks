#include "unique_finder.cuh"
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
__global__ void count_occurrences_kernel(T* data, int* histogram, size_t n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&histogram[data[index]], 1);
    }
}

template <typename T>
UniqueFinder<T>::UniqueFinder(const std::vector<T>& data, size_t nunique) {
    data_size = data.size();
    this->nunique = nunique;

    cudaError_t err = cudaMalloc(&d_data, data_size * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "Error during cudaMalloc: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_histogram, nunique * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error during cudaMalloc: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(d_data, data.data(), data_size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error during cudaMemcpy: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemset(d_histogram, 0, nunique * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error during cudaMemset: " << cudaGetErrorString(err) << std::endl;
    }
}

template <typename T>
UniqueFinder<T>::~UniqueFinder() {
    cudaFree(d_data);
    cudaFree(d_histogram);
}

template <typename T>
std::vector<T> UniqueFinder<T>::find_unique() {
    err = count_occurrences_kernel<<<(data_size + 255) / 256, 256>>>(d_data, d_histogram, data_size);
    if (err != cudaSuccess) {
        std::cerr << "Error during kernel execution: " << cudaGetErrorString(err) << std::endl;
    }

    int* h_histogram = new int[nunique];
    err = cudaMemcpy(h_histogram, d_histogram, nunique * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error during cudaMemcpy: " << cudaGetErrorString(err) << std::endl;
    }

    std::vector<T> unique_values;
    for (size_t i = 0; i < nunique; i++) {
        std::cout << i << ": " << h_histogram[i] << std::endl;
        if (h_histogram[i] == 1) {
            unique_values.push_back(static_cast<T>(i));
        }
    }

    delete[] h_histogram;
    return unique_values;
}
