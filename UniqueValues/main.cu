#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <fstream>
#include <set>
#include <unordered_map>
#include <cuda_runtime.h>
//#include "unique_finder.cuh"

enum Constants {
    N = 1000,       // Number of random integers
    T = 10,          // Number of trials
    UNIQUE_VALUES = 700,  // limiting to 1000 unique values
};

std::vector<int> generate_random_numbers(int n, int unique_values) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, unique_values - 1);

    std::vector<int> numbers;
    for (int i = 0; i < n; ++i) {
        numbers.push_back(dis(gen));
    }

    return numbers;
}

double time_function(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

double get_median(std::vector<double>& times) {
    int size = times.size();
    sort(times.begin(), times.end());
    if (size % 2 == 0) {
        return (times[size / 2 - 1] + times[size / 2]) / 2;
    } else {
        return times[size / 2];
    }
}

std::string toJsonString(const std::vector<std::pair<std::string, std::string>>& items) {
    std::string result = "{\n";
    for (const auto& item : items) {
        result += "    \"" + item.first + "\": " + item.second + ",\n";
    }
    result.pop_back(); // Remove the last comma
    result.pop_back(); 
    result += "\n}";
    return result;
}

bool check_correctness(const std::vector<int>& original_data, const std::vector<int>& found_unique_elements) {
    std::unordered_map<int, int> elements_counts;
    for (auto val: original_data) {
        elements_counts[val]++;
    }
    std::cout << "True histogram:" << std::endl;
    std::vector<int> true_unique_values;
    for (auto& el : elements_counts) {
        int key = el.first;
        int value = el.second;
        std::cout << key << ": " << value << std::endl;
        if (value == 1) {
            true_unique_values.push_back(key);
        }
    }
    std::cout << "True unique values: ";
    for (auto val : true_unique_values) {
        std::cout << val << " ";
    }
    std::cout << std::endl << "Found unique values: ";
    for (auto val : found_unique_elements) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    if (true_unique_values.size() != found_unique_elements.size()) {
        return false;
    }
    return std::set<int>(true_unique_values.begin(), true_unique_values.end()) == 
            std::set<int>(found_unique_elements.begin(), found_unique_elements.end());
}

__global__ void count_occurrences_kernel(int* data, int* histogram, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        atomicAdd(&histogram[data[index]], 1);
    }
}

int main() {
    std::vector<int> data = generate_random_numbers(N, UNIQUE_VALUES);
    int n = data.size();
    int nunique = UNIQUE_VALUES;

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
        std::cout << i << " " << h_histogram[i] << std::endl;
        //if (h_histogram[i] == 1) {
        //    std::cout << i << " ";
        //}
    }

    cudaFree(d_data);
    cudaFree(d_histogram);
    delete[] h_histogram;

    return 0;
}
