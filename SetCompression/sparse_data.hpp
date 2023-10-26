#pragma once

#include "data_format.hpp"
#include <vector>

template <typename T>
class SparseData : public DataFormat<T> {
public:
    // Constructs the SparseData object using a vector of pairs
    explicit SparseData(std::vector<std::pair<T, T>>&& sparse_data);

    std::vector<T> get_data() const override;
    size_t get_size() const override;

private:
    std::vector<std::pair<T, T>> sparse_data_;  // Pair structure: (index, value)
};

// Explicit template instantiation for common types
template class SparseData<int>;
template class SparseData<long long>;
