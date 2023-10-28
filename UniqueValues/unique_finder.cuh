#pragma once

#include <vector>

template <typename T>
class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder(const std::vector<T>& data, int nunique);
    ~UniqueFinder();

    std::vector<T> find_unique();

private:
    T* d_data = nullptr;
    int* d_histogram = nullptr;
    int data_size = 0;
    int nunique = 0;
};

// Explicit template instantiation for common types
template class UniqueFinder<int>;
