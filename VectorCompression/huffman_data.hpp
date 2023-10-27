#pragma once

#include "data_format.hpp"
#include <vector>

template <typename T>
class HuffmanData : public DataFormat<T> {
public:
    explicit HuffmanData(std::vector<T>&& dense_array, size_t original_size);

    std::vector<T> get_data() const override;
    size_t get_size() const override;

private:
    std::vector<T> dense_array_;
    size_t original_size_;
};

// Explicit template instantiation for common types
template class HuffmanData<int>;
template class HuffmanData<long long>;
