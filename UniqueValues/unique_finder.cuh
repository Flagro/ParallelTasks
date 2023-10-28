#pragma once

#include <vector>

template <typename T>
class UniqueFinder {
public:
    UniqueFinder() = default;
    UniqueFinder(std::vector<T>&& data, size_t nunique);
    ~UniqueFinder();

    std::vector<T> findUnique();
private:
    T* d_data_;
    T* d_unique_values_;
    T* d_histogram_;
    std::vector<T> histogram_;
    size_t unique_values_;
};

// Explicit template instantiation for common types
template class UniqueFinder<int>;
template class UniqueFinder<long long>;
