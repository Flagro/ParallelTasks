#pragma once

#include "data_format.hpp"
#include <vector>

template <typename T>
class BitsetData : public DataFormat<T> {
public:
    explicit BitsetData(const std::vector<T>& input_array, size_t max_value);

    std::vector<T> get_data() const override;
    size_t get_size() const override;

private:
    std::vector<bool> compressed_data_;
    int bits_per_number_; // Number of bits required for each number
};

// Explicit template instantiation for common types
template class BitsetData<int>;
template class BitsetData<long long>;
