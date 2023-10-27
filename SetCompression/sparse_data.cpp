#include "omp.h"
#include "sparse_data.hpp"

template <typename T>
SparseData<T>::SparseData(std::vector<std::pair<T, T>>&& sparse_data, size_t original_size)
    : sparse_data_(std::move(sparse_data)), original_size_(original_size) {}

template <typename T>
std::vector<T> SparseData<T>::get_data() const {
    std::vector<T> reconstructed_data;
    for (const auto& entry : sparse_data_) {
        for (T i = 0; i < entry.second; ++i) {
            reconstructed_data.push_back(entry.first);
        }
    }
    return reconstructed_data;
}

template <typename T>
size_t SparseData<T>::get_size() const {
    return sparse_data_.capacity() * (2 * sizeof(T));  // Capacity of pairs (index, value)
}
