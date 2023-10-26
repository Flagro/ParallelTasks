#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include "compressed_data.hpp"

enum Constants {
    N = 1000000,       // Number of random integers
    T = 1000,          // Number of trials
    MAX_VALUE = 100    // Max value for random numbers
};

std::vector<int> generate_random_numbers(int n, int max_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, max_value);

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
    size_t size = times.size();
    sort(times.begin(), times.end());
    if (size % 2 == 0) {
        return (times[size / 2 - 1] + times[size / 2]) / 2;
    } else {
        return times[size / 2];
    }
}

int main() {
    std::vector<double> compression_times, get_data_times, get_size_times;
    std::vector<size_t> sizes;

    for (int trial = 0; trial < T; ++trial) {
        std::vector<int> numbers = generate_random_numbers(N, MAX_VALUE);

        // Measure compression time
        compression_times.push_back(time_function([&]() { CompressedData<int> compressed(numbers, MAX_VALUE); }));

        CompressedData<int> compressed(numbers, MAX_VALUE);

        // Measure get_data time
        get_data_times.push_back(time_function([&]() { auto data = compressed.get_data(); }));

        // Measure get_size time
        get_size_times.push_back(time_function([&]() { auto size = compressed.get_size(); }));

        sizes.push_back(compressed.get_size());
    }

    std::cout << "Median Compression Time: " << get_median(compression_times) << " ms" << std::endl;
    std::cout << "Median GetData Time: " << get_median(get_data_times) << " ms" << std::endl;
    std::cout << "Median GetSize Time: " << get_median(get_size_times) << " ms" << std::endl;
    std::cout << "Min Size: " << *std::min_element(sizes.begin(), sizes.end()) << " bytes" << std::endl;
    std::cout << "Max Size: " << *std::max_element(sizes.begin(), sizes.end()) << " bytes" << std::endl;
    std::cout << "Mean Size: " << std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size() << " bytes" << std::endl;

    return 0;
}
