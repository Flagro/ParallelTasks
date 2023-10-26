#include "dense_data.hpp"

template <typename T>
DenseData<T>::DenseData(std::vector<T>&& dense_array)
    : dense_array_(std::move(dense_array)) {}

template <typename T>
std::vector<T> DenseData<T>::get_data() const {
    std::vector<T> original_data;
    for (T i = 0; i < dense_array_.size(); ++i) {
        for (T j = 0; j < dense_array_[i]; ++j) {
            original_data.push_back(i);
        }
    }
    return original_data;
}

template <typename T>
size_t DenseData<T>::get_size() const {
    return sizeof(T) * dense_array_.capacity();
}
