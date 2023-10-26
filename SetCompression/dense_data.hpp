#pragma once

#include "data_format.hpp"
#include <vector>

template <typename T>
class DenseData : public DataFormat<T> {
public:
    explicit DenseData(std::vector<T>&& dense_array);

    std::vector<T> get_data() const override;
    size_t get_size() const override;

private:
    std::vector<T> dense_array_;
};
