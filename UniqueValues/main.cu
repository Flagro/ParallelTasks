#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_occurrences_kernel(int* data, int* histogram, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&histogram[data[index]], 1);
    }
}

int main() {
    std::vector<int> data = {1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 5};
    int n = data.size();
    int nunique = 10;

    int* d_data;
    int* d_histogram;
    
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_histogram, nunique * sizeof(int));
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, nunique * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    count_occurrences_kernel<<<blocks, threadsPerBlock>>>(d_data, d_histogram, n);

    int* h_histogram = new int[nunique];
    cudaMemcpy(h_histogram, d_histogram, nunique * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nunique; i++) {
        if (h_histogram[i] == 1) {
            std::cout << i << " ";
        }
    }

    cudaFree(d_data);
    cudaFree(d_histogram);
    delete[] h_histogram;

    return 0;
}
