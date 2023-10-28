#include "compressed_data.hpp"
#include "huffman_data.hpp"
// #include "bitset_data.hpp"
#include <vector>
#include <omp.h>

template <typename T>
CompressedData<T>::CompressedData(const std::vector<T>& input, T max_value) {
    data_format_ = std::make_unique<HuffmanData<T>>(input, max_value);
}

template <typename T>
std::vector<T> CompressedData<T>::get_data() const {
    return data_format_->get_data();
}

template <typename T>
size_t CompressedData<T>::get_size() const {
    return data_format_->get_size();
}
