#include "compressed_data.hpp"
#include "sparse_data.hpp"
#include "dense_data.hpp"
#include <vector>
#include <omp.h>

template <typename T>
CompressedData<T>::CompressedData(const std::vector<T>& input, T max_value) {
    
}

template <typename T>
std::vector<T> CompressedData<T>::get_data() const {
    return data_format_->get_data();
}

template <typename T>
size_t CompressedData<T>::get_size() const {
    return data_format_->get_size();
}
