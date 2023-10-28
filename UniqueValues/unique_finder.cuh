#pragma once

#include <vector>

template <typename T>
class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder(const std::vector<T>& data, size_t nunique);
    ~UniqueFinder();

    std::vector<T> find_unique();

private:
    T* d_data = nullptr;                 // Device pointer for input data
    int* d_histogram = nullptr;          // Device pointer for histogram
    T* d_unique_values = nullptr;        // Device pointer for storing unique values
    int* d_unique_counter = nullptr;     // Device pointer for counting unique values
    size_t data_size;                    // Size of the input data
    size_t nunique;                      // Number of unique values
};

// Explicit template instantiation for common types
template class UniqueFinder<int>;
