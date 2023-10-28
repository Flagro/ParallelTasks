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
    T* d_data = nullptr;
    int* d_histogram = nullptr;
    size_t data_size = 0;
    size_t nunique = 0;

    __global__ void count_occurrences_kernel(T* data, int* histogram, size_t n);
};

// Explicit template instantiation for common types
template class UniqueFinder<int>;
