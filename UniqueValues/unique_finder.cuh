#pragma once

#include <vector>
#include <cuda_runtime.h>

template <typename T>
class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder(const std::vector<T>& data, size_t nunique);
    ~UniqueFinder();

    std::vector<T> find_unique();

private:
    const T* d_data = nullptr;  // Pointer to device data
    T* d_histogram = nullptr;   // Pointer to histogram on device
    size_t nunique_;
    size_t data_size_;
};

// Explicit template instantiation for common types
template class UniqueFinder<int>;
