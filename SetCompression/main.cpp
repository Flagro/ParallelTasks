#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <set>
#include "omp.h"
#include "compressed_data.hpp"

enum Constants {
    N = 1000000,       // Number of random integers
    T = 100,          // Number of trials
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

int main() {
    std::ofstream json_file("results.json");
    std::string json_results = "[\n";
    
    omp_set_dynamic(1);
    const auto omp_max_threads = omp_get_num_threads();

    for (int num_threads = 1; num_threads <= omp_max_threads; ++num_threads) {
        omp_set_num_threads(num_threads);
        
        std::cout << "Processing with " << num_threads << " threads..." << std::endl;

        std::vector<double> compression_times, get_data_times, get_size_times;
        std::vector<size_t> sizes;

        for (int trial = 0; trial < T; ++trial) {
            std::vector<int> numbers = generate_random_numbers(N, MAX_VALUE);

            CompressedData<int> compressed;

            compression_times.push_back(time_function([&]() { compressed = CompressedData<int>(numbers, MAX_VALUE); }));

            get_data_times.push_back(time_function([&]() { auto data = compressed.get_data(); }));
            get_size_times.push_back(time_function([&]() { auto size = compressed.get_size(); }));
            sizes.push_back(compressed.get_size());

            // JSON output
            std::vector<std::pair<std::string, std::string>> run_results;
            run_results.emplace_back("threads", std::to_string(num_threads));
            run_results.emplace_back("data_size", std::to_string(N));
            run_results.emplace_back("data_max_value", std::to_string(MAX_VALUE));
            run_results.emplace_back("data_nunique", std::to_string(std::set<int>(numbers.begin(), numbers.end()).size()));
            run_results.emplace_back("compression_time", std::to_string(compression_times.back()));
            run_results.emplace_back("get_data_time", std::to_string(get_data_times.back()));
            run_results.emplace_back("get_size_time", std::to_string(get_size_times.back()));
            run_results.emplace_back("compressed_size", std::to_string(sizes.back()));
            run_results.emplace_back("original_size", std::to_string(numbers.capacity() * sizeof(int)));

            json_results += toJsonString(run_results) + ",\n";
        }

        // Print the results to stdout
        std::cout << "Threads: " << num_threads << std::endl;
        std::cout << "Median Compression Time: " << get_median(compression_times) << " ms" << std::endl;
        std::cout << "Median GetData Time: " << get_median(get_data_times) << " ms" << std::endl;
        std::cout << "Median GetSize Time: " << get_median(get_size_times) << " ms" << std::endl;
        std::cout << "Min Size: " << *std::min_element(sizes.begin(), sizes.end()) << " bytes" << std::endl;
        std::cout << "Max Size: " << *std::max_element(sizes.begin(), sizes.end()) << " bytes" << std::endl;
        std::cout << "Mean Size: " << std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size() << " bytes" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    json_results.pop_back(); // Remove the last comma
    json_results.pop_back(); 
    json_results += "\n]";

    json_file << json_results;
    json_file.close();

    return 0;
}
