#include "omp.h"
#include "dense_data.hpp"

template <typename T>
DenseData<T>::DenseData(std::vector<T>&& dense_array, size_t original_size)
    : dense_array_(std::move(dense_array)), original_size_(original_size) {}

template <typename T>
std::vector<T> DenseData<T>::get_data() const {
    std::vector<T> original_data(dense_array_.size());
    #pragma omp parallel for
    for (T i = 0; i < dense_array_.size(); ++i) {
        size_t pos = 0;  // Starting position for this value
        for (T j = 0; j < i; ++j) {
            pos += dense_array_[j];
        }
        for (T j = 0; j < dense_array_[i]; ++j) {
            original_data[pos + j] = i;
        }
    }
    return original_data;
}

template <typename T>
size_t DenseData<T>::get_size() const {
    return sizeof(T) * dense_array_.capacity();
}
