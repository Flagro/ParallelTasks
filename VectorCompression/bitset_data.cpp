#include "bitset_data.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
BitsetData<T>::BitsetData(const std::vector<T>& input_array, size_t max_value) {
    bits_per_number_ = std::ceil(std::log2(max_value + 1)); // calculate bits needed
    compressed_data_.resize(input_array.size() * bits_per_number_);

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (size_t idx = 0; idx < input_array.size(); idx++) {
            T num = input_array[idx];
            size_t startPos = idx * bits_per_number_;
            for (int i = bits_per_number_ - 1; i >= 0; --i) {
                compressed_data_[startPos + bits_per_number_ - 1 - i] = (num >> i) & 1;
            }
        }
    }
}

template <typename T>
std::vector<T> BitsetData<T>::get_data() const {
    std::vector<T> decompressed_data;
    decompressed_data.resize(compressed_data_.size() / bits_per_number_);

    #pragma omp parallel for
    for (size_t i = 0; i < compressed_data_.size(); i += bits_per_number_) {
        T num = 0;
        for (int j = bits_per_number_ - 1; j >= 0; --j) {
            num |= (compressed_data_[i + bits_per_number_ - 1 - j] << j);
        }
        decompressed_data[i / bits_per_number_] = num;
    }

    return decompressed_data;
}

template <typename T>
size_t BitsetData<T>::get_size() const {
    return compressed_data_.size() / 8 + (compressed_data_.size() % 8 != 0); // Convert from bits to bytes
}
