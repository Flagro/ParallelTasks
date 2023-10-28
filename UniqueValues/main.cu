#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <fstream>
#include <set>
#include <unordered_map>
#include "unique_finder.cuh"

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

    std::vector<int> true_unique_values;
    for (auto& el : elements_counts) {
        int key = el.first;
        int value = el.second;
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

int main() {
    std::ofstream json_file("results.json");
    std::string json_results = "[\n";

    std::vector<double> allocation_times, get_unique_times, get_intimes;
    std::vector<int> sizes;
    bool correctness = true;

    std::cout << "Started running..." << std::endl;
    for (int trial = 0; trial < T; ++trial) {
        std::vector<int> data = generate_random_numbers(N, UNIQUE_VALUES);

        UniqueFinder<int> finder;

        allocation_times.push_back(time_function([&]() { finder = UniqueFinder<int>(data, UNIQUE_VALUES); }));

        std::vector<int> unique_elements;

        get_unique_times.push_back(time_function([&]() { unique_elements = finder.find_unique(); }));

        bool is_correct = check_correctness(data, unique_elements);

        if (!is_correct) {
            correctness = false;
            std::cout << "Incorrect result!" << std::endl;
        }

        // JSON output
        std::vector<std::pair<std::string, std::string>> run_results;
        run_results.emplace_back("data_size", std::to_string(N));
        run_results.emplace_back("data_nunique", std::to_string(UNIQUE_VALUES));
        run_results.emplace_back("allocation_time", std::to_string(allocation_times.back()));
        run_results.emplace_back("get_unique_time", std::to_string(get_unique_times.back()));
        run_results.emplace_back("correct", std::to_string(is_correct));

        json_results += toJsonString(run_results) + ",\n";
    }

    // Print the results to stdout
    std::cout << "Median Allocation Time: " << get_median(allocation_times) << " ms" << std::endl;
    std::cout << "Median GetUnique Time: " << get_median(get_unique_times) << " ms" << std::endl;
    std::cout << "Data Length: " << N << std::endl;
    std::cout << "Data UniqueCount: " << UNIQUE_VALUES << std::endl;
    std::cout << "Correctness: " << correctness << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    json_results.pop_back(); // Remove the last comma
    json_results.pop_back(); 
    json_results += "\n]";

    json_file << json_results;
    json_file.close();

    return 0;
}

