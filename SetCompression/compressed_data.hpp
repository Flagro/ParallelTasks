#pragma once

#include "data_format.hpp"
#include <vector>
#include <memory>

template <typename T>
class CompressedData {
private:
    std::unique_ptr<DataFormat<T>> data_format_;

public:
    CompressedData(const std::vector<T>& input, T max_value);
    std::vector<T> get_data() const;
    size_t get_size() const;
};
